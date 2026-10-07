/**
 * pi-sandbox-bridge — Pi extension 转发 8 工具到 Poirot SpecialistMcpServer.
 *
 * 决策 1（设计文档 46 §10.5）：强制走 Poirot SpecialistMcpServer。
 * pi --no-builtin-tools 禁用自带 read/write/edit/bash，
 * 此 extension 注册 8 个 poirot_* 工具，全部通过 stdio MCP 转发到
 * Poirot SpecialistMcpServer（POIROT_SANDBOX_MCP_ENDPOINT env 指定）。
 *
 * 经过 PathTranslator + SecurityGuard（与 codex/claude 统一沙箱访问）。
 * 物理无隔离，pi 写的文件 lead agent 立即可见。
 *
 * 注意：此文件放在 Python 仓库（Poirot）内，但运行时由 pi 用 jiti 加载。
 * jiti 跳过 tsc 类型检查，所以 // @ts-nocheck 不影响运行时。
 * 去掉 typebox 依赖（pi.registerTool 接受 plain JSON Schema object）。
 */

// @ts-nocheck — pi 用 jiti 加载，跳过 tsc 类型检查；Poirot 仓库无 npm 依赖

// 8 个工具名 + 参数 schema（与 Poirot SpecialistMcpServer 暴露的接口一致）
// 用 plain JSON Schema object，不依赖 @sinclair/typebox
const POIROT_TOOLS = [
  {
    name: "bash",
    description: "Execute a bash command in the Poirot sandbox (security-guarded).",
    parameters: {
      type: "object",
      properties: {
        command: { type: "string", description: "The bash command to execute." },
      },
      required: ["command"],
    },
  },
  {
    name: "read_file",
    description: "Read a file from the Poirot sandbox (security-guarded).",
    parameters: {
      type: "object",
      properties: {
        path: { type: "string", description: "File path to read." },
      },
      required: ["path"],
    },
  },
  {
    name: "write_file",
    description: "Write content to a file in the Poirot sandbox (security-guarded).",
    parameters: {
      type: "object",
      properties: {
        path: { type: "string", description: "File path to write." },
        content: { type: "string", description: "Content to write." },
      },
      required: ["path", "content"],
    },
  },
  {
    name: "list_dir",
    description: "List directory contents in the Poirot sandbox (security-guarded).",
    parameters: {
      type: "object",
      properties: {
        path: { type: "string", description: "Directory path to list." },
      },
      required: ["path"],
    },
  },
  {
    name: "str_replace",
    description: "Replace string in a file in the Poirot sandbox (security-guarded).",
    parameters: {
      type: "object",
      properties: {
        path: { type: "string", description: "File path." },
        old_str: { type: "string", description: "String to replace." },
        new_str: { type: "string", description: "Replacement string." },
      },
      required: ["path", "old_str", "new_str"],
    },
  },
  {
    name: "glob",
    description: "Find files matching glob pattern in the Poirot sandbox (security-guarded).",
    parameters: {
      type: "object",
      properties: {
        pattern: { type: "string", description: "Glob pattern." },
        path: { type: "string", description: "Base path inside the sandbox." },
      },
      required: ["pattern", "path"],
    },
  },
  {
    name: "grep",
    description: "Search file contents with regex in the Poirot sandbox (security-guarded).",
    parameters: {
      type: "object",
      properties: {
        pattern: { type: "string", description: "Regex pattern." },
        path: { type: "string", description: "File or directory path." },
      },
      required: ["pattern", "path"],
    },
  },
  {
    name: "download_file",
    description: "Download a file to the Poirot sandbox (security-guarded).",
    parameters: {
      type: "object",
      properties: {
        url: { type: "string", description: "URL to download." },
        path: { type: "string", description: "Existing file path inside the sandbox to read/download." },
      },
      required: ["path"],
    },
  },
];

/**
 * Extension entry point.
 *
 * pi 加载此 extension 后，agent 看到 poirot_bash / poirot_read_file / poirot_write_file 等 8 个工具。
 * 所有工具通过 stdio MCP 协议调 Poirot SpecialistMcpServer。
 */
export default function (pi) {
  for (const toolDef of POIROT_TOOLS) {
    const toolName = toolDef.name;
    const poirotToolName = `poirot_${toolName}`;
    pi.registerTool({
      name: poirotToolName,
      label: `Poirot ${toolName}`,
      description: toolDef.description,
      parameters: toolDef.parameters,
      execute: async (_toolCallId, params) => {
        const result = await callPoirotMcp(toolName, params);
        return {
          content: [{ type: "text", text: result }],
          details: {},
        };
      },
    });
  }
}

/**
 * 通过 stdio MCP 协议调 Poirot SpecialistMcpServer.
 *
 * POIROT_SANDBOX_MCP_ENDPOINT env 由 PiRuntime._build_env 注入，
 * 格式：python -m poirot.backend.agents.multiagent.mcp.specialist_mcp_server --sandbox-id <id>
 *
 * 此函数启动 SpecialistMcpServer 子进程，通过 stdio MCP JSON-RPC 调用指定 tool。
 *
 * MVP：每次调用启动新子进程（与 codex/claude 的 SpecialistMcpServer 调用模式一致）。
 * 进阶：可改为持久 MCP client 连接。
 */
async function callPoirotMcp(tool, args) {
  const endpoint = process.env.POIROT_SANDBOX_MCP_ENDPOINT;
  if (!endpoint) {
    throw new Error(
      "POIROT_SANDBOX_MCP_ENDPOINT not set. PiRuntime._build_env should inject it."
    );
  }

  // Endpoint is emitted by Python with shell-free arguments encoded as a
  // JSON array when possible. Keep the legacy string form for compatibility,
  // but never split a quoted Windows path blindly.
  let cmdParts;
  try {
    const parsed = JSON.parse(endpoint);
    cmdParts = Array.isArray(parsed) ? parsed : String(endpoint).trim().split(/\s+/);
  } catch {
    cmdParts = String(endpoint).trim().split(/\s+/);
  }

  // 启动 SpecialistMcpServer 子进程
  const { spawn } = require("child_process");
  const proc = spawn(cmdParts[0], cmdParts.slice(1), {
    stdio: ["pipe", "pipe", "pipe"],
  });

  return new Promise((resolve, reject) => {
    let stdoutBuffer = "";
    let stderrBuffer = "";
    let initialized = false;
    let sentCall = false;

    proc.stdout.on("data", (data) => {
      stdoutBuffer += data.toString();
      const lines = stdoutBuffer.split("\n");
      stdoutBuffer = lines.pop() || "";
      for (const line of lines) {
        if (!line.trim()) continue;
        let msg;
        try { msg = JSON.parse(line); } catch { continue; }
        if (msg.id === 0 && msg.result && !initialized) {
          initialized = true;
          proc.stdin.write(JSON.stringify({
            jsonrpc: "2.0",
            method: "notifications/initialized",
            params: {},
          }) + "\n");
          proc.stdin.write(JSON.stringify({
            jsonrpc: "2.0",
            id: 1,
            method: "tools/call",
            params: { name: tool, arguments: args },
          }) + "\n");
          sentCall = true;
        }
        if (msg.id === 1 && sentCall) {
          if (msg.error) {
            reject(new Error(`MCP error: ${msg.error.message || JSON.stringify(msg.error)}`));
          } else {
            if (msg.result?.isError) {
              const errorText = (msg.result?.content || [])
                .filter((c) => c.type === "text")
                .map((c) => c.text)
                .join("\n");
              reject(new Error(`Sandbox tool ${tool} failed: ${errorText || "unknown error"}`));
              proc.stdin.end();
              proc.kill();
              return;
            }
            const textParts = (msg.result?.content || [])
              .filter((c) => c.type === "text")
              .map((c) => c.text);
            // Keep a compact machine-readable marker in the tool result so
            // Poirot can distinguish a successful empty result from a failed
            // specialist response without exposing file contents.
            resolve(`[POIROT_TOOL_OK:${tool}] ${textParts.join("\n") || "(empty result)"}`);
          }
          proc.stdin.end();
          proc.kill();
        }
      }
    });

    proc.stderr.on("data", (data) => {
      stderrBuffer += data.toString();
    });

    proc.on("error", (err) => {
      reject(new Error(`SpecialistMcpServer spawn failed: ${err.message}`));
    });

    proc.on("close", (code) => {
      if (code !== 0) {
        reject(
          new Error(
            `SpecialistMcpServer exited with code ${code}: ${stderrBuffer}`
          )
        );
        return;
      }
      // 尝试从 stdout 解析 MCP JSON-RPC response
      // MCP stdio 协议：每行一个 JSON-RPC message
      const lines = stdoutBuffer.split("\n").filter((l) => l.trim());
      for (const line of lines) {
        try {
          const msg = JSON.parse(line);
          // JSON-RPC response 含 result 或 error
          if (msg.result && msg.result.content) {
            if (msg.result.isError) {
              const errorText = msg.result.content
                .filter((c) => c.type === "text")
                .map((c) => c.text)
                .join("\n");
              reject(new Error(`Sandbox tool ${tool} failed: ${errorText || "unknown error"}`));
              return;
            }
            // MCP tool result: { content: [{ type: "text", text: "..." }] }
            const textParts = msg.result.content
              .filter((c) => c.type === "text")
              .map((c) => c.text);
            resolve(`[POIROT_TOOL_OK:${tool}] ${textParts.join("\n") || "(empty result)"}`);
            return;
          }
          if (msg.error) {
            reject(new Error(`MCP error: ${msg.error.message || JSON.stringify(msg.error)}`));
            return;
          }
        } catch {
          // 非 JSON 行跳过
          continue;
        }
      }
      // A process closing without the matching tools/call response is not a
      // successful tool invocation. Do not turn protocol noise into evidence.
      reject(new Error(
        `SpecialistMcpServer returned no response for ${tool}` +
        (stderrBuffer ? `: ${stderrBuffer}` : "")
      ));
    });

    // MCP stdio is line-delimited JSON in the Python SDK used by Poirot.
    // Complete initialize first; the data handler then sends the required
    // notifications/initialized and tools/call messages.
    const initRequest = {
      jsonrpc: "2.0",
      id: 0,
      method: "initialize",
      params: {
        protocolVersion: "2024-11-05",
        capabilities: {},
        clientInfo: { name: "pi-sandbox-bridge", version: "0.1.0" },
      },
    };
    const callRequest = {
      jsonrpc: "2.0",
      id: 1,
      method: "tools/call",
      params: {
        name: tool,
        arguments: args,
      },
    };
    proc.stdin.write(JSON.stringify(initRequest) + "\n");
  });
}
