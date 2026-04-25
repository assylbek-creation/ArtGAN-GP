"""Smoke tests so pytest has something to run end-to-end."""


def test_imports() -> None:
    import src  # noqa: F401
    import src.data  # noqa: F401
    import src.models  # noqa: F401
    import src.training  # noqa: F401
    import src.utils  # noqa: F401
