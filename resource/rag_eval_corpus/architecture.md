# Poirot architecture

Poirot uses one LeaderAgent to orchestrate the research loop. LangGraph
orchestrates the outer flow: prepare, leader_agent, and finalize. Middleware
crosses lifecycle hooks such as before_agent, before_model, and wrap_tool_call.

The context governance layer uses compaction and externalization to avoid
context overflow. Historical messages may be summarized or written outside the
prompt while keeping a preview and a path.

MCP supports stdio, sse, and http transports. Tool fallback can route
web_search from an MCP server to the builtin ddg implementation.
