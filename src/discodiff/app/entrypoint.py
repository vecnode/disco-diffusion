"""Thin entrypoint delegating to the notebook-style runtime in :mod:`discodiff.main`."""


def run(cli_overrides: dict | None = None):
    """Same as :func:`discodiff.main.main` (stable alias for packaging / tests)."""
    from ..main import main as main_impl

    return main_impl(cli_overrides=cli_overrides)


