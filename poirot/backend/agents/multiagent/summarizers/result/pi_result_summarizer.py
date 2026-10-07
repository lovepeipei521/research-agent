"""PiResultSummarizer — extends base with three-section parsing (What You Did / Success / Gaps).

设计（design_docs/46 §10.3.6）:
- 继承 BaseResultSummarizer 通用 programmatic eval floor
- Pi-specific：解析 pi 输出的三段（_build_prompt 要求的 What You Did / Success / Gaps）
- 不传回 raw output 全量（控制 lead agent context 膨胀）
"""
from __future__ import annotations

import re

from poirot.backend.agents.multiagent.summarizers.result.base import (
    BaseResultSummarizer,
)
from poirot.backend.agents.multiagent.types import ArtifactRef


# 三段输出格式正则（与 PiRuntime._build_prompt 的 Output Format 一致）
_WHAT_YOU_DID_RE = re.compile(
    r"##\s*What You Did\s*\n(.*?)(?=##\s*Success|##\s*Gaps|$)",
    re.DOTALL | re.IGNORECASE,
)
_SUCCESS_SECTION_RE = re.compile(
    r"##\s*Success\s*\n(.*?)(?=##\s*Gaps|$)",
    re.DOTALL | re.IGNORECASE,
)
_GAPS_SECTION_RE = re.compile(
    r"##\s*Gaps\s*\n(.*?)$",
    re.DOTALL | re.IGNORECASE,
)
# success 判定关键词（yes / pass / met）
_SUCCESS_POSITIVE_RE = re.compile(r"\b(yes|pass|met|complete|success|succeeded)\b", re.IGNORECASE)
# success 判定关键词（no / fail / not met / partial）
_SUCCESS_NEGATIVE_RE = re.compile(r"\b(no|fail|not met|partial|incomplete)\b", re.IGNORECASE)


class PiResultSummarizer(BaseResultSummarizer):
    """Pi specialist 输出端转换器（三段解析 + programmatic eval floor）。

    继承 BaseResultSummarizer 通用校验（产物存在性 + success_criteria 回应 + 无敏感改动）。
    Pi-specific 扩展：解析 _build_prompt 要求的三段输出格式。
    """

    def __init__(self) -> None:
        super().__init__(specialist_name="pi")

    def summarize(
        self,
        raw_output: str,
        artifacts: list[ArtifactRef],
        goal: str,
        success_criteria: str,
        evidence: list[str] | None = None,
    ) -> "SpecialistResult":  # type: ignore[name-defined]
        """压缩 raw output + 校验 success_criteria + 解析三段。"""
        # 调 base 通用校验（产物存在性 + success_criteria 回应 + 无敏感改动）
        base_success = self._evaluate_success(raw_output, artifacts, success_criteria, evidence)

        # 解析 pi 输出的三段
        sections = self._parse_pi_sections(raw_output)

        # Deterministic task/evidence evaluation. The Success prose is only a
        # secondary negative signal, never the primary success source.
        pi_success = self._evaluate_pi_success(
            sections, base_success, raw_output, success_criteria, evidence or []
        )

        # gap_analysis：优先用 Gaps 段，否则用 base 提取
        gap = sections.get("gaps", "") or (
            "" if pi_success else self._extract_gap(raw_output, success_criteria)
        )

        # summary：优先用 What You Did 段，否则用 base 压缩
        summary = sections.get("did", "") or self._compress(raw_output)

        from poirot.backend.agents.multiagent.types import SpecialistResult

        return SpecialistResult(
            specialist_name=self._specialist_name,
            summary=summary,
            artifacts=tuple(artifacts),
            success=pi_success,
            gap_analysis=gap,
            execution_status="completed" if raw_output.strip() else "failed",
            criteria_status=(
                "met" if pi_success else ("not_met" if raw_output.strip() else "unknown")
            ),
        )

    def _evaluate_success(
        self,
        raw_output: str,
        artifacts: list[ArtifactRef],
        success_criteria: str,
        evidence: list[str] | None = None,
    ) -> bool:
        # Pi's sandbox bridge reports successful tool calls as evidence. This
        # is sufficient for a write/read/edit operation even when the runtime
        # does not create a lead-agent artifact record.
        evidence = evidence or []
        if any("error" in item.lower() or "failed" in item.lower() for item in evidence):
            return False
        if not raw_output.strip() or self._has_sensitive_changes(raw_output):
            return False
        criteria_lower = success_criteria.lower()
        no_file_answer = (
            "只返回" in criteria_lower
            or "only return" in criteria_lower
            or "do not read, modify, or create" in criteria_lower
        ) and (
            "不要读取、修改或创建" in criteria_lower
            or "不要读取" in criteria_lower
            or "no file" in criteria_lower
            or "without file" in criteria_lower
        )
        if no_file_answer:
            # A literal answer task is successful with non-empty output and no
            # explicit failure, even though it intentionally has no artifact
            # and no sandbox tool call.
            lowered = raw_output.lower()
            if any(marker in lowered for marker in ("specialistemptyoutput", "failed", "error:", "not met", "incomplete")):
                return False
            return True
        tool_calls = [item[5:] for item in evidence if item.startswith("tool:")]
        if any(tool.startswith("error:") for tool in tool_calls):
            return False
        negative_artifact = any(
            marker in criteria_lower
            for marker in ("不要文件", "不创建文件", "不修改文件", "不要创建任何文件", "no file", "without file")
        )
        requires_artifact = any(
            marker in criteria_lower
            for marker in ("create file", "write file", "save file", "generate report", "artifact", "写入文件", "创建文件", "生成报告", "产物")
        )
        if not negative_artifact and requires_artifact and not artifacts and not tool_calls:
            return False
        if not negative_artifact and not requires_artifact and not artifacts and not tool_calls:
            # Non-artifact Pi tasks are valid only when the caller explicitly
            # says it is read-only/no-file work; this avoids accepting an
            # arbitrary unverified answer as a completed coding task.
            read_only = any(marker in criteria_lower for marker in ("只读", "不要读取、修改或创建", "不要修改", "不要创建", "read-only", "without modifying"))
            if not read_only and "只返回" not in criteria_lower:
                return False
        return True

    def _parse_pi_sections(self, output: str) -> dict[str, str]:
        """解析 pi 输出的 What You Did / Success / Gaps 三段。

        pi 按 _build_prompt 要求的格式输出，此方法提取三段内容。
        缺失段返空串。
        """
        sections: dict[str, str] = {}

        did_match = _WHAT_YOU_DID_RE.search(output)
        if did_match:
            sections["did"] = did_match.group(1).strip()

        success_match = _SUCCESS_SECTION_RE.search(output)
        if success_match:
            sections["success"] = success_match.group(1).strip()

        gaps_match = _GAPS_SECTION_RE.search(output)
        if gaps_match:
            sections["gaps"] = gaps_match.group(1).strip()

        return sections

    def _evaluate_pi_success(
        self,
        sections: dict[str, str],
        base_success: bool,
        raw_output: str = "",
        success_criteria: str = "",
        evidence: list[str] | None = None,
    ) -> bool:
        """pi-specific success 判定。

        判定逻辑：
        1. base 通用校验必须通过（产物存在 + success_criteria 回应 + 无敏感改动）
        2. Success 段无负面关键词（no / fail / not met / partial / incomplete）
        3. 如果 Success 段有正面关键词（yes / pass / met / complete），加强信心
        """
        if not base_success:
            return False

        evidence = evidence or []
        criteria = success_criteria.lower()
        output = raw_output.lower()
        explicit_failure = any(
            marker in output
            for marker in ("specialistemptyoutput", "tool failed", "not met", "incomplete", "unable to complete")
        )
        if explicit_failure:
            return False

        no_file_answer = (
            ("只返回" in criteria or "only return" in criteria)
            and any(marker in criteria for marker in ("不要读取", "不要修改", "不要创建", "no file", "without file"))
        )
        if no_file_answer:
            return bool(raw_output.strip())

        tool_success = any(item.startswith("tool:") for item in evidence)
        requires_operation = any(
            marker in criteria
            for marker in ("写入", "创建", "修改", "替换", "write", "create", "modify", "replace", "test", "测试")
        )
        if requires_operation and not tool_success:
            return False

        success_section = sections.get("success", "").lower()
        if not success_section:
            # Simple confirmation/read-only requests may explicitly require a
            # literal one-line response, so requiring Pi's three-section
            # template would contradict the task contract.
            return base_success

        # 有负面关键词 → 失败
        if _SUCCESS_NEGATIVE_RE.search(success_section):
            return False

        # 无负面关键词 → 通过（base 已校验）
        return True
