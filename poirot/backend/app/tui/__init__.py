"""Poirot TUI — 全屏终端应用（textual 驱动）。

与 ``app/cli/``（传统滚动 CLI）并行。``poirot chat`` 默认启动 TUI；
现有 CLI 路径可通过 ``--legacy`` 参数回退。

复用 ``app/services/stream_service.py`` 的 ``PoirotStreamClient`` + ``StreamEvent``
数据层，仅替换呈现层：StreamEvent → textual Widget 渲染。
"""

from poirot.backend.app.tui.app import PoirotTUI

__all__ = ["PoirotTUI"]
