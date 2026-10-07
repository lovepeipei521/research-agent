"""Batch B5 self-check: McpLoader (no real server, just structural)."""
import asyncio
from poirot.backend.agents.mcp.config import McpConfig, McpServerConfig
from poirot.backend.agents.mcp.loader import McpLoader
from poirot.backend.agents.mcp.registry import ToolRegistry
from langchain_core.tools import tool


def test_empty_config():
    """空 config → load_startup 返空列表，不抛。"""
    cfg = McpConfig()
    reg = ToolRegistry(cfg)
    loader = McpLoader(cfg, reg)
    tools = asyncio.run(loader.load_startup())
    assert tools == [], f"expected empty, got {tools}"
    print("PASS: empty config → no tools, no error")


def test_disabled_server_skipped():
    """enabled=false 的 server 被跳过。"""
    cfg = McpConfig(servers={
        "test": McpServerConfig(name="test", transport="stdio", command="echo", enabled=False),
    })
    reg = ToolRegistry(cfg)
    loader = McpLoader(cfg, reg)
    tools = asyncio.run(loader.load_startup())
    assert tools == [], "disabled server should be skipped"
    print("PASS: disabled server skipped")


def test_failed_server_not_blocking():
    """单 server 连接失败不阻塞（command 不存在会失败）。"""
    cfg = McpConfig(servers={
        "bad": McpServerConfig(
            name="bad", transport="stdio", command="nonexistent_cmd_xyz",
            connect_timeout=2,
        ),
    })
    reg = ToolRegistry(cfg)
    loader = McpLoader(cfg, reg)
    tools = asyncio.run(loader.load_startup())
    # 失败 server 不返工具，但也不抛
    assert tools == [], "failed server should not return tools"
    print("PASS: failed server not blocking")


def test_shutdown_safe():
    """shutdown 无 server 时安全。"""
    cfg = McpConfig()
    reg = ToolRegistry(cfg)
    loader = McpLoader(cfg, reg)
    asyncio.run(loader.shutdown())
    print("PASS: shutdown safe (no clients)")


def test_env_filter_applied():
    """EnvFilter guard 在 guards 列表内。"""
    cfg = McpConfig()
    reg = ToolRegistry(cfg)
    loader = McpLoader(cfg, reg)
    assert any(type(g).__name__ == "EnvFilter" for g in loader._guards)
    print("PASS: EnvFilter in guards")


def test_sanitizer_exposed():
    """sanitizer 属性可访问。"""
    cfg = McpConfig()
    reg = ToolRegistry(cfg)
    loader = McpLoader(cfg, reg)
    assert loader.sanitizer is not None
    assert loader.sanitizer.sanitize_error("ghp_test") == loader.sanitizer.sanitize_error("ghp_test")
    print("PASS: sanitizer exposed")


def test_core_tools_env_override(tmp_path, monkeypatch):
    """POIROT_MCP_CORE_TOOLS 应覆盖 YAML 的 core_tools。"""
    from poirot.backend.agents.mcp.config import load_mcp_config

    path = tmp_path / "mcp.yaml"
    path.write_text("core_tools: [web_search]\n", encoding="utf-8")
    monkeypatch.setenv("POIROT_MCP_CORE_TOOLS", "web_search, browse_page")
    cfg = load_mcp_config(str(path))
    assert cfg.core_tools == ["web_search", "browse_page"]
    print("PASS: env core_tools override")


def test_server_include_filters_dynamic_tools():
    """MCP include 白名单应阻止未测试的动态工具进入 registry。"""
    @tool("browse_page")
    def browse_page(url: str) -> str:
        """Fetch one page."""
        return url

    @tool("parallel_browse")
    def parallel_browse(urls: list[str]) -> str:
        """Fetch several pages."""
        return str(urls)

    cfg = McpConfig()
    registry = ToolRegistry(cfg)
    loader = McpLoader(cfg, registry)
    server = McpServerConfig(
        name="freeweb",
        transport="stdio",
        command="npx",
        include_tools=["browse_page"],
    )
    registered = loader._register_tools(server, [browse_page, parallel_browse])
    assert [item.name for item in registered] == ["browse_page"]
    assert registry.get("parallel_browse") is None
    print("PASS: MCP include filter")


if __name__ == "__main__":
    test_empty_config()
    test_disabled_server_skipped()
    test_failed_server_not_blocking()
    test_shutdown_safe()
    test_env_filter_applied()
    test_sanitizer_exposed()
    print("\nAll B5 self-checks passed.")
