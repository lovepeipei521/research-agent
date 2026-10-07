import tomllib


def test_pyproject_exposes_poirot_console_script() -> None:
    with open("pyproject.toml", "rb") as file:
        pyproject = tomllib.load(file)

    assert pyproject["project"]["scripts"]["poirot"] == "poirot.backend.app.cli.main:main"
