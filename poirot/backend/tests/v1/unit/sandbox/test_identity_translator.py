from __future__ import annotations

from poirot.backend.agents.sandbox.contracts import PathTranslator
from poirot.backend.agents.sandbox.translators.identity_translator import (
    IdentityTranslator,
)


class TestIdentityTranslator:
    def test_translate_path_passthrough(self) -> None:
        translator = IdentityTranslator()
        assert translator.translate_path("/mnt/poirot/user-data/x") == "/mnt/poirot/user-data/x"

    def test_translate_command_passthrough(self) -> None:
        translator = IdentityTranslator()
        assert translator.translate_command("ls /mnt/poirot/user-data") == "ls /mnt/poirot/user-data"

    def test_mask_output_passthrough(self) -> None:
        translator = IdentityTranslator()
        assert translator.mask_output("output") == "output"

    def test_is_path_translator(self) -> None:
        translator = IdentityTranslator()
        assert isinstance(translator, PathTranslator)

    def test_empty_path(self) -> None:
        translator = IdentityTranslator()
        assert translator.translate_path("") == ""

    def test_complex_command(self) -> None:
        translator = IdentityTranslator()
        cmd = "find /mnt/poirot/user-data -name '*.py' | xargs grep 'pattern'"
        assert translator.translate_command(cmd) == cmd
