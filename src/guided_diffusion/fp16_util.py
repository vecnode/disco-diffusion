"""Minimal float16/float32 conversion helpers used by UNet runtime."""

import torch.nn as nn


def convert_module_to_f16(module):
    """Convert primitive convolution modules to float16."""
    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        module.weight.data = module.weight.data.half()
        if module.bias is not None:
            module.bias.data = module.bias.data.half()


def convert_module_to_f32(module):
    """Convert primitive convolution modules back to float32."""
    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        module.weight.data = module.weight.data.float()
        if module.bias is not None:
            module.bias.data = module.bias.data.float()
