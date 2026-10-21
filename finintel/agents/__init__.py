"""Agentic layer: Claude tool-calling analyst with structured output, plus orchestration."""
from .claude_agent import DEFAULT_MODEL, ClaudeAdvisoryAgent, merge_narrative
from .orchestrator import FinancialIntelligencePipeline, PipelineConfig
from .tools import TOOL_DEFS, AgentToolbox

__all__ = ["DEFAULT_MODEL", "TOOL_DEFS", "AgentToolbox", "ClaudeAdvisoryAgent", "FinancialIntelligencePipeline", "PipelineConfig", "merge_narrative"]
