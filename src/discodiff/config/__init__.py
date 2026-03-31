"""Run configuration: ``SimpleNamespace`` built from ``main()`` locals."""

from .keyframes import get_inbetweens, parse_key_frames, split_prompts
from .run_args import build_run_args_namespace

__all__ = [
    "build_run_args_namespace",
    "get_inbetweens",
    "parse_key_frames",
    "split_prompts",
]
