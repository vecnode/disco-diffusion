"""Thin entrypoint delegating to the notebook-style runtime in :mod:`run`."""


def run(cli_overrides: dict | None = None):
    """Same as :func:`run.main` (stable alias for packaging / tests)."""
    from run import main as main_impl

    return main_impl(cli_overrides=cli_overrides)


