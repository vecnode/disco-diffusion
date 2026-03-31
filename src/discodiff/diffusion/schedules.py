"""DDIM / training step count derived from user ``steps`` (notebook-compatible)."""


def timestep_respacing_ddim(steps: int) -> str:
    """``timestep_respacing`` string for guided-diffusion (e.g. ``ddim250``)."""
    return f"ddim{steps}"


def diffusion_steps_count(steps: int) -> int:
    """Base training step count paired with respacing (legacy formula from ``main``)."""
    return (1000 // steps) * steps if steps < 1000 else steps
