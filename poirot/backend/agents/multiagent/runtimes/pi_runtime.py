"""PiRuntime — RPC mode 协议实现（pi --mode rpc 子进程 + stdio JSON-RPC）。

设计（spec.md PiRuntime Requirement + design_docs/46 §10.3）:
- sync only MVP（INV#5，同 CodexRuntime/ClaudeCodeRuntime）
- 每次 invoke 启动新 pi 子进程 + 完成关闭（不做 pool）
- 传 goal + context_summary + success_criteria（决策 5：MVP 不定制 system prompt，
  靠 _build_prompt 在 user message 塞 context + 三段输出格式要求）
- 超时 kill + crash SpecialistCrashError
- pi 黑盒：pi CLI 自带 model + 自管 ReAct loop（INV#1/INV#2）
- 决策 1：强制走 Poirot SpecialistMcpServer（--no-builtin-tools + -e poirot-sandbox-bridge）
- 决策 3：凭证 env 透传（国内 provider 优先）
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import time
import re
import threading
import queue
from dataclasses import dataclass
from typing import Any

from poirot.backend.agents.multiagent.exceptions import (
    SpecialistCrashError,
    SpecialistError,
    SpecialistStartupError,
    SpecialistTimeoutError,
)
from poirot.backend.agents.multiagent.runtimes import append_sandbox_url_args
from poirot.backend.agents.multiagent.types import (
    SpecialistRawResult,
    SpecialistRequest,
    TokenUsage,
)

logger = logging.getLogger(__name__)
_TOOL_EVIDENCE_RE = re.compile(r"\[POIROT_TOOL_OK:([A-Za-z0-9_]+)\]")


def _read_stderr(stream: Any) -> str:
    """Read real subprocess stderr and lightweight iterator-based test stubs."""
    if stream is None:
        return ""
    reader = getattr(stream, "read", None)
    if callable(reader):
        value = reader()
        return value if isinstance(value, str) else str(value or "")
    try:
        return "".join(str(item) for item in stream)
    except TypeError:
        return str(stream)


@dataclass(frozen=True)
class PiRuntimeConfig:
    """Pi runtime 配置（决策 5：不定制 system prompt，靠 _build_prompt）。

    provider/model/thinking_level 留空时用 pi 默认（pi 自己选）。
    """

    command: str = "pi"
    mode_args: tuple[str, ...] = ("--mode", "rpc", "--no-session")
    provider: str | None = None
    model: str | None = None
    thinking_level: str = "medium"
    extra_args: tuple[str, ...] = ()


class PiRuntime:
    """Pi coding agent runtime via RPC mode (stdio JSON-RPC).

    sync only MVP（同 CodexRuntime/ClaudeCodeRuntime）。
    每次 invoke 启动新 pi 子进程 + 完成关闭。

    决策 1（设计文档 46 §10.5）：强制走 Poirot SpecialistMcpServer——
    --no-builtin-tools 禁用 pi 自带 read/write/edit/bash，
    -e poirot-sandbox-bridge 加载 Poirot sandbox bridge extension，
    所有操作走 Poirot MCP 8 接口（经过 PathTranslator + SecurityGuard）。
    """

    def __init__(self, config: PiRuntimeConfig | None = None, sandbox_provider=None) -> None:
        self._config = config or PiRuntimeConfig()
        self._sandbox_provider = sandbox_provider
        self._credential = None

    def set_credential(self, credential: Any | None) -> None:
        """Attach the credential selected by Poirot bootstrap without logging it."""
        self._credential = credential

    def invoke(self, request: SpecialistRequest) -> SpecialistRawResult:
        start = time.time()
        try:
            raw_output, usage, evidence = self._run_rpc_session(request)
        except subprocess.TimeoutExpired:
            raise SpecialistTimeoutError(
                timeout_seconds=request.timeout_seconds,
            )
        except FileNotFoundError:
            raise SpecialistStartupError(
                f"pi command not found: {self._config.command}. "
                "Install: npm install -g @earendil-works/pi-coding-agent"
            )
        except SpecialistError:
            raise
        except Exception as e:
            raise SpecialistCrashError(str(e))

        return SpecialistRawResult(
            raw_output=raw_output,
            usage=usage,
            duration_seconds=time.time() - start,
            evidence=tuple(evidence),
        )

    def _run_rpc_session(
        self, request: SpecialistRequest
    ) -> tuple[str, TokenUsage | None, list[str]]:
        """启动 pi --mode rpc 子进程，发 prompt，收 events，返 final text + usage。"""
        cmd = self._build_command(request)
        # Windows npm global executables are usually ``pi.cmd``.  CreateProcess
        # does not reliably resolve a bare ``pi`` entry from a list-form argv,
        # even though ``shutil.which`` and an interactive CMD can find it.
        resolved_command = shutil.which(self._config.command)
        if resolved_command is None and os.name == "nt":
            resolved_command = shutil.which(f"{self._config.command}.cmd")
        if resolved_command is None:
            raise FileNotFoundError(self._config.command)
        cmd[0] = resolved_command
        env = self._build_env(request)

        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        try:
            prompt = self._build_prompt(request)
            cmd_json = json.dumps(
                {"type": "prompt", "message": prompt, "id": "req-1"}
            )
            proc.stdin.write(cmd_json + "\n")
            proc.stdin.flush()

            chunks: list[str] = []
            usage_data: dict[str, Any] = {}
            evidence: list[str] = []
            saw_agent_end = False
            # stdout iteration is blocking on Windows pipes. A watchdog makes
            # a stuck Pi/MCP chain fail deterministically instead of hanging
            # the TUI until the user interrupts it.
            deadline = time.monotonic() + max(1, request.timeout_seconds)
            lines: queue.Queue[Any] = queue.Queue()

            def _read_stdout() -> None:
                try:
                    for item in proc.stdout:
                        lines.put(item)
                finally:
                    lines.put(None)

            reader = threading.Thread(target=_read_stdout, daemon=True)
            reader.start()
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise SpecialistTimeoutError(timeout_seconds=request.timeout_seconds)
                line = lines.get(timeout=remaining)
                if line is None:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue

                etype = event.get("type")
                if etype == "response" and event.get("success") is False:
                    error = event.get("error") or "Pi RPC prompt failed"
                    raise SpecialistStartupError(f"Pi RPC error: {error}")
                if etype == "message_update":
                    ame = event.get("assistantMessageEvent", {})
                    if ame.get("type") == "text_delta":
                        chunks.append(ame["delta"])
                    # Pi emits tool calls/results as assistant message events.
                    # Preserve only names/status and paths, never full content.
                    if ame.get("type") in {"tool_call", "tool_result"}:
                        tool_name = ame.get("toolName") or ame.get("name") or ame.get("tool_name")
                        if tool_name:
                            evidence.append(f"tool:{tool_name}")
                        result = ame.get("result")
                        if isinstance(result, str) and result.strip():
                            evidence.append(f"tool_result:{result[:500]}")

                elif etype == "agent_end":
                    saw_agent_end = True
                    # 从最后一条 assistant message 提取 usage
                    msgs = event.get("messages", [])
                    for msg in reversed(msgs):
                        if msg.get("role") == "assistant":
                            usage_data = msg.get("usage", {}) or {}
                            break
                    break

                elif etype == "extension_error":
                    raise SpecialistCrashError(
                        f"pi extension error: {event.get('error', 'unknown')}"
                    )

            raw_output = "".join(chunks)
            # Bridge markers are intentionally compact and contain no file
            # contents. They are evidence for the result evaluator only.
            evidence.extend(f"tool:{name}" for name in _TOOL_EVIDENCE_RE.findall(raw_output))
            raw_output = _TOOL_EVIDENCE_RE.sub("", raw_output)
            returncode = proc.poll()
            if not raw_output and isinstance(returncode, int) and returncode != 0:
                stderr = _read_stderr(proc.stderr).strip()
                raise SpecialistCrashError(
                    f"pi exited with code {returncode}: {stderr or 'no output'}"
                )
            if not raw_output and not usage_data and not saw_agent_end:
                stderr = _read_stderr(proc.stderr).strip()
                raise SpecialistCrashError(
                    "pi returned no assistant output (RPC session ended without agent_end)"
                    + (f"; stderr: {stderr}" if stderr else "")
                )
            usage = self._extract_usage(usage_data)
            return raw_output, usage, evidence

        finally:
            self._cleanup_process(proc)

    @staticmethod
    def _cleanup_process(proc: Any) -> None:
        """Best-effort idempotent cleanup for Pi and its Windows wrapper tree."""
        try:
            if proc.stdin is not None:
                proc.stdin.close()
        except (OSError, ValueError):
            pass
        try:
            if proc.poll() is None:
                proc.terminate()
        except (OSError, ValueError):
            pass
        try:
            proc.wait(timeout=2)
        except Exception:
            try:
                proc.kill()
            except (OSError, ValueError):
                pass
        # ``pi.cmd`` can leave a child node process behind on Windows. Kill
        # only the known process tree rooted at this PID when available.
        pid = getattr(proc, "pid", None)
        if os.name == "nt" and isinstance(pid, int) and pid > 0:
            try:
                subprocess.run(
                    ["taskkill", "/PID", str(pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                    timeout=2,
                )
            except (OSError, subprocess.TimeoutExpired):
                pass

    def _build_command(self, request: SpecialistRequest) -> list[str]:
        """组装 pi CLI 命令（决策 1：--no-builtin-tools + -e poirot-sandbox-bridge）。"""
        cmd = [self._config.command, *self._config.mode_args]
        if self._config.provider:
            cmd.extend(["--provider", self._config.provider])
        model = self._config.model
        if self._config.provider == "deepseek" and model == "deepseek-v4.1-flash":
            model = "deepseek-flash"
        if model:
            cmd.extend(["--model", model])
        if self._config.thinking_level:
            cmd.extend(["--thinking", self._config.thinking_level])
        # 决策 1：禁用 pi 自带工具，加载 Poirot sandbox bridge extension
        cmd.extend(["--no-builtin-tools"])
        cmd.extend(["-e", str(self._poirot_extension_path())])
        # 决策 5：不传 --system-prompt（MVP 靠 _build_prompt 在 user message 塞 context）
        cmd.extend(self._config.extra_args)
        return cmd

    def _build_env(self, request: SpecialistRequest) -> dict[str, str] | None:
        """透传凭证 env vars + Poirot sandbox MCP endpoint（决策 1 + 决策 3）。

        决策 3：凭证 env 国内 provider 优先（DeepSeek/Kimi/MiniMax/Xiaomi 靠前）。
        决策 1：传 POIROT_SANDBOX_MCP_ENDPOINT（sandbox_id 绑定）给 pi extension。
        返 merge 后的 env（父 env + auth vars 覆盖），保证 PATH/HOME 等基础 env 可用。
        """
        auth_env: dict[str, str] = {}
        # 国内 provider 优先（决策 3，便宜优先）
        pi_env_vars = [
            "DEEPSEEK_API_KEY",
            "KIMI_API_KEY",
            "MINIMAX_API_KEY",
            "XIAOMI_API_KEY",
            "ZAI_API_KEY",
            # 国外大厂
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "GEMINI_API_KEY",
            # 聚合/其他
            "OPENROUTER_API_KEY",
            "GROQ_API_KEY",
            "XAI_API_KEY",
            "MISTRAL_API_KEY",
            "TOGETHER_API_KEY",
        ]
        for var in pi_env_vars:
            val = os.getenv(var)
            if val:
                auth_env[var] = val

        # PiCredential may come from POIROT_MULTIAGENT_PI_API_KEY rather than
        # a provider-specific environment variable. Map it to the provider
        # variable Pi itself reads.
        credential_key = getattr(self._credential, "api_key", None)
        if credential_key:
            provider = getattr(self._credential, "provider", None) or self._config.provider
            provider_env = {
                "deepseek": "DEEPSEEK_API_KEY",
                "anthropic": "ANTHROPIC_API_KEY",
                "openai": "OPENAI_API_KEY",
                "google": "GEMINI_API_KEY",
                "kimi-coding": "KIMI_API_KEY",
                "minimax": "MINIMAX_API_KEY",
                "xiaomi": "XIAOMI_API_KEY",
                "zai": "ZAI_API_KEY",
            }.get(provider)
            auth_env[provider_env or "PI_API_KEY"] = credential_key

        # PI_CODING_AGENT_DIR（自定义配置目录）
        pi_dir = os.getenv("PI_CODING_AGENT_DIR")
        # Pi writes a lock beside settings.json on every startup.  A broken
        # user-level directory must not prevent Poirot from delegating; use a
        # project-local writable directory by default and preserve an explicit
        # user override.
        if pi_dir:
            try:
                os.makedirs(pi_dir, exist_ok=True)
            except OSError as exc:
                logger.warning("Pi config directory is not writable: %s (%s)", pi_dir, exc)
            auth_env["PI_CODING_AGENT_DIR"] = pi_dir
        elif request.sandbox_id:
            # Only specialist sessions need a private writable Pi config dir;
            # preserve the old no-auth/no-sandbox env contract for callers and
            # unit tests that only inspect environment forwarding.
            pi_dir = str(self._pi_config_dir())
            try:
                os.makedirs(pi_dir, exist_ok=True)
                auth_env["PI_CODING_AGENT_DIR"] = pi_dir
            except OSError as exc:
                logger.warning("Unable to create project Pi config directory: %s", exc)

        # 决策 1：传 Poirot SpecialistMcpServer endpoint（sandbox_id 绑定）
        if request.sandbox_id:
            auth_env["POIROT_SANDBOX_MCP_ENDPOINT"] = self._resolve_mcp_endpoint(
                request.sandbox_id
            )

        if not auth_env:
            return None
        # merge：父进程 env + auth vars 覆盖（保证 PATH/HOME 等基础 env 可用）
        return {**os.environ, **auth_env}

    def _build_prompt(self, request: SpecialistRequest) -> str:
        """构造给 pi 的 prompt（决策 5：MVP 不定制 system prompt）。

        pi 用自己的默认 system prompt（通用 coding agent）。
        Poirot 在 user message 里塞 goal + context_summary + success_criteria + 三段输出格式要求。
        """
        parts = [request.goal]
        if request.context_summary:
            parts.append(f"\n\n## Context\n{request.context_summary}")
        if request.success_criteria:
            parts.append(f"\n\n## Success Criteria\n{request.success_criteria}")
        parts.append(
            "\n\n## Output Format\n"
            "After completing the task, summarize in three sections:\n"
            "## What You Did\n"
            "- Files changed (paths + brief description)\n"
            "- Commands run\n"
            "## Success\n"
            "- Whether success criteria are met (yes/no/partial)\n"
            "- Evidence (test results, file existence, etc.)\n"
            "## Gaps\n"
            "- Any incomplete parts or blockers\n"
            "- Next steps if incomplete"
        )
        return "".join(parts)

    def _extract_usage(
        self, usage_data: dict[str, Any]
    ) -> TokenUsage | None:
        """从 pi agent_end 事件的 usage 字段提取 TokenUsage。"""
        if not usage_data:
            return None
        try:
            tokens = usage_data.get("tokens") or {}
            return TokenUsage(
                prompt_tokens=int(tokens.get("input", 0)),
                completion_tokens=int(tokens.get("output", 0)),
                total_tokens=int(tokens.get("total", 0)),
            )
        except (TypeError, ValueError):
            return None

    def _resolve_mcp_endpoint(self, sandbox_id: str) -> str:
        """解析 Poirot SpecialistMcpServer endpoint（sandbox_id 绑定）。

        MVP：返回 SpecialistMcpServer 的 stdio 启动命令（pi extension 通过此命令连接）。
        块 D3：若 sandbox_provider 有 SandboxInfo，追加 --sandbox-url + --sandbox-root。
        """
        # Use the interpreter hosting Poirot.  On Windows, a bare ``python``
        # can resolve to a different installation than the active venv.
        parts = [
            sys.executable,
            "-m",
            "poirot.backend.agents.multiagent.mcp.specialist_mcp_server",
            "--sandbox-id",
            sandbox_id,
        ]
        append_sandbox_url_args(parts, self._sandbox_provider, sandbox_id)
        # JSON array avoids corrupting Windows paths containing spaces.
        return json.dumps(parts, ensure_ascii=False)

    def _poirot_extension_path(self) -> str:
        """Poirot sandbox bridge extension 路径（随 Poirot 安装）。

        决策 1：pi extension 注册 8 工具转发到 Poirot SpecialistMcpServer。
        extension 文件在 P4 batch 创建（当前路径返回预期位置，extension 不存在时
        pi 会报 extension_error，由 _run_rpc_session 捕获为 SpecialistCrashError）。
        """
        from poirot.backend.agents.multiagent import __path__ as pkg_path

        import pathlib

        return str(
            pathlib.Path(pkg_path[0])
            / "extensions"
            / "pi-sandbox-bridge"
            / "index.ts"
        )

    def _pi_config_dir(self):
        from pathlib import Path
        return Path.cwd() / ".poirot" / "pi-agent"
