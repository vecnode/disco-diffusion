"""Diffusion sampling loop entry. Delegates to the nested `do_run` published on `discodiff.main`."""


def invoke_diffusion() -> None:
    """Run CLIP-guided diffusion; implementation lives in `main.main` to preserve closure semantics."""
    import discodiff.main as _main

    _main.do_run()
