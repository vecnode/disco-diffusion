"""Continuous-domain resize for NumPy and PyTorch.

Implements resizing via weighted samples over a projected 1D grid per dimension,
with optional antialiasing on downscale. Based on the approach in "From Discrete
to Continuous Convolutions" (Shocher et al.).
"""
from __future__ import annotations

import warnings
from fractions import Fraction
from math import ceil, pi


try:
    import numpy
except ImportError:
    warnings.warn("NumPy not found; only PyTorch tensors are supported.", stacklevel=1)
    numpy = None

try:
    import torch
except ImportError:
    warnings.warn("PyTorch not found; only NumPy arrays are supported.", stacklevel=1)
    torch = None

if numpy is None and torch is None:
    raise ImportError("Either NumPy or PyTorch is required.")


def _framework_eps_and_cast(x):
    if numpy is not None and isinstance(x, numpy.ndarray):
        framework = numpy
        to_dtype = lambda a: a  # noqa: E731
    else:
        framework = torch
        to_dtype = lambda a: a.to(x.dtype)  # noqa: E731
    eps = framework.finfo(framework.float32).eps
    return framework, to_dtype, eps


def _with_support_size(width: float):
    def decorator(fn):
        fn.support_sz = width
        return fn

    return decorator


@_with_support_size(4)
def cubic(x):
    """Bicubic interpolation kernel (Keys)."""
    fw, to_dtype, eps = _framework_eps_and_cast(x)
    absx = fw.abs(x)
    absx2 = absx**2
    absx3 = absx**3
    return (1.5 * absx3 - 2.5 * absx2 + 1.0) * to_dtype(absx <= 1.0) + (
        -0.5 * absx3 + 2.5 * absx2 - 4.0 * absx + 2.0
    ) * to_dtype((1.0 < absx) & (absx <= 2.0))


@_with_support_size(4)
def lanczos2(x):
    fw, to_dtype, eps = _framework_eps_and_cast(x)
    return (
        (fw.sin(pi * x) * fw.sin(pi * x / 2) + eps)
        / ((pi**2 * x**2 / 2) + eps)
    ) * to_dtype(abs(x) < 2)


@_with_support_size(6)
def lanczos3(x):
    fw, to_dtype, eps = _framework_eps_and_cast(x)
    return (
        (fw.sin(pi * x) * fw.sin(pi * x / 3) + eps)
        / ((pi**2 * x**2 / 3) + eps)
    ) * to_dtype(abs(x) < 3)


@_with_support_size(2)
def linear(x):
    fw, to_dtype, eps = _framework_eps_and_cast(x)
    return (x + 1) * to_dtype((-1 <= x) & (x < 0)) + (1 - x) * to_dtype(
        (0 <= x) & (x <= 1)
    )


@_with_support_size(1)
def box(x):
    fw, to_dtype, eps = _framework_eps_and_cast(x)
    return to_dtype((-1 <= x) & (x < 0)) + to_dtype((0 <= x) & (x <= 1))


def resize(
    arr,
    scale_factors=None,
    out_shape=None,
    interp_method=cubic,
    support_sz=None,
    antialiasing=True,
    by_convs=False,
    scale_tolerance=None,
    max_numerator=10,
    pad_mode="constant",
):
    """Resize ``arr`` along each dimension where scale differs from 1.

    ``arr`` must be a NumPy ndarray or a PyTorch tensor; the same type is
    returned. Either ``scale_factors`` or ``out_shape`` (or both) must be given.
    """
    in_shape, n_dims = arr.shape, arr.ndim
    fw = numpy if (numpy is not None and isinstance(arr, numpy.ndarray)) else torch
    eps = fw.finfo(fw.float32).eps
    device = arr.device if fw is torch else None

    scale_factors, out_shape, by_convs = set_scale_and_out_sz(
        in_shape,
        out_shape,
        scale_factors,
        by_convs,
        scale_tolerance,
        max_numerator,
        eps,
        fw,
    )

    dims_to_resize = [
        (
            dim,
            scale_factors[dim],
            by_convs[dim],
            in_shape[dim],
            out_shape[dim],
        )
        for dim in sorted(range(n_dims), key=lambda i: scale_factors[i])
        if scale_factors[dim] != 1.0
    ]

    if support_sz is None:
        support_sz = interp_method.support_sz

    out = arr
    for dim, scale_factor, dim_by_convs, in_sz, out_sz in dims_to_resize:
        projected_grid = get_projected_grid(
            in_sz, out_sz, scale_factor, fw, dim_by_convs, device
        )
        cur_interp, cur_support = apply_antialiasing_if_needed(
            interp_method, support_sz, scale_factor, antialiasing
        )
        field_of_view = get_field_of_view(
            projected_grid, cur_support, fw, eps, device
        )
        pad_spec, projected_grid, field_of_view = calc_pad_sz(
            in_sz,
            out_sz,
            field_of_view,
            projected_grid,
            scale_factor,
            dim_by_convs,
            fw,
            device,
        )
        weights = get_weights(cur_interp, projected_grid, field_of_view)

        if not dim_by_convs:
            out = apply_weights(
                out, field_of_view, weights, dim, n_dims, pad_spec, pad_mode, fw
            )
        else:
            out = apply_convs(
                out,
                scale_factor,
                in_sz,
                out_sz,
                weights,
                dim,
                pad_spec,
                pad_mode,
                fw,
            )
    return out


def get_projected_grid(in_sz, out_sz, scale_factor, fw, by_convs, device=None):
    """Map output pixel centers to continuous coordinates on the input axis."""
    grid_sz = out_sz if not by_convs else scale_factor.numerator
    out_coords = fw_arange(grid_sz, fw, device)
    sf = float(scale_factor)
    return (
        out_coords / sf
        + (in_sz - 1) / 2
        - (out_sz - 1) / (2 * sf)
    )


def get_field_of_view(projected_grid, cur_support_sz, fw, eps, device):
    left = fw_ceil(projected_grid - cur_support_sz / 2 - eps, fw)
    ordinal = fw_arange(ceil(cur_support_sz - eps), fw, device)
    return left[:, None] + ordinal


def calc_pad_sz(
    in_sz, out_sz, field_of_view, projected_grid, scale_factor, dim_by_convs, fw, device
):
    if not dim_by_convs:
        pad_spec = [
            -field_of_view[0, 0].item(),
            field_of_view[-1, -1].item() - in_sz + 1,
        ]
        field_of_view += pad_spec[0]
        projected_grid += pad_spec[0]
    else:
        num_convs, stride = scale_factor.numerator, scale_factor.denominator
        left_pads = -field_of_view[:, 0]
        right_pads = (
            (out_sz - fw_arange(num_convs, fw, device) - 1) // num_convs
        ) * stride + field_of_view[:, -1] - in_sz + 1
        pad_spec = list(zip(left_pads, right_pads))

    return pad_spec, projected_grid, field_of_view


def get_weights(interp_method, projected_grid, field_of_view):
    delta = projected_grid[:, None] - field_of_view
    weights = interp_method(delta)
    sum_w = weights.sum(1, keepdims=True)
    sum_w[sum_w == 0] = 1
    return weights / sum_w


def apply_weights(arr, field_of_view, weights, dim, n_dims, pad_sz, pad_mode, fw):
    tmp = fw_swapaxes(arr, dim, 0, fw)
    tmp = fw_pad(tmp, fw, pad_sz, pad_mode)
    neighbors = tmp[field_of_view]
    w_expanded = fw.reshape(
        weights, (*weights.shape, *([1] * (n_dims - 1)))
    )
    out = (neighbors * w_expanded).sum(1)
    return fw_swapaxes(out, 0, dim, fw)


def apply_convs(arr, scale_factor, in_sz, out_sz, weights, dim, pad_sz, pad_mode, fw):
    x = fw_swapaxes(arr, dim, -1, fw)
    stride, num_convs = scale_factor.denominator, scale_factor.numerator
    shape_out = list(x.shape)
    shape_out[-1] = out_sz
    out = fw_empty(tuple(shape_out), fw, x.device)

    pad_dim = x.ndim - 1
    for conv_ind, (lr_pad, filt) in enumerate(zip(pad_sz, weights)):
        x_padded = fw_pad(x, fw, lr_pad, pad_mode, dim=pad_dim)
        out[..., conv_ind::num_convs] = fw_conv1d(x_padded, filt, stride)
    return fw_swapaxes(out, -1, dim, fw)


def set_scale_and_out_sz(
    in_shape, out_shape, scale_factors, by_convs, scale_tolerance, max_numerator, eps, fw
):
    if scale_factors is None and out_shape is None:
        raise ValueError("Provide scale_factors and/or out_shape.")

    if out_shape is not None:
        if fw is numpy:
            out_shape = list(out_shape) + list(in_shape[len(out_shape) :])
        else:
            out_shape = list(in_shape[: -len(out_shape)]) + list(out_shape)
        if scale_factors is None:
            scale_factors = [o / i for o, i in zip(out_shape, in_shape)]

    if scale_factors is not None:
        if not isinstance(scale_factors, (list, tuple)):
            scale_factors = [scale_factors, scale_factors]
        if fw is numpy:
            scale_factors = list(scale_factors) + [1.0] * (
                len(in_shape) - len(scale_factors)
            )
        else:
            scale_factors = [1.0] * (len(in_shape) - len(scale_factors)) + list(
                scale_factors
            )
        if out_shape is None:
            out_shape = [
                ceil(sf * ins) for sf, ins in zip(scale_factors, in_shape)
            ]

        if not isinstance(by_convs, (list, tuple)):
            by_convs = [by_convs] * len(out_shape)

        for ind, (sf, use_convs) in enumerate(zip(scale_factors, by_convs)):
            frac = None
            if use_convs:
                inv = Fraction(1 / sf).limit_denominator(max_numerator)
                frac = Fraction(numerator=inv.denominator, denominator=inv.numerator)

            tol = scale_tolerance if scale_tolerance is not None else eps
            if use_convs and abs(frac - sf) < tol:
                scale_factors[ind] = frac
            else:
                scale_factors[ind] = float(sf)
                by_convs[ind] = False

        return scale_factors, out_shape, by_convs


def apply_antialiasing_if_needed(interp_method, support_sz, scale_factor, antialiasing):
    sf = float(scale_factor)
    if sf >= 1.0 or not antialiasing:
        return interp_method, support_sz
    stretched = lambda arg, im=interp_method, s=sf: s * im(s * arg)  # noqa: E731
    return stretched, support_sz / sf


def fw_ceil(x, fw):
    return fw.int_(fw.ceil(x)) if fw is numpy else x.ceil().long()


def fw_swapaxes(x, ax_1, ax_2, fw):
    return fw.swapaxes(x, ax_1, ax_2) if fw is numpy else x.transpose(ax_1, ax_2)


def fw_pad(x, fw, pad_sz, pad_mode, dim=0):
    if pad_sz == (0, 0):
        return x
    if fw is numpy:
        pad_vec = [(0, 0)] * x.ndim
        pad_vec[dim] = pad_sz
        return fw.pad(x, pad_width=pad_vec, mode=pad_mode)

    if x.ndim < 3:
        x = x[None, None, Ellipsis]
    pad_vec = [0] * ((x.ndim - 2) * 2)
    pad_vec[0:2] = pad_sz
    return fw.nn.functional.pad(
        x.transpose(dim, -1), pad=pad_vec, mode=pad_mode
    ).transpose(dim, -1)


def fw_conv1d(arr, weight_1d, stride):
    """1D convolution along the last dimension (PyTorch only)."""
    reshaped = arr.reshape(1, 1, -1, arr.shape[-1])
    conv_out = torch.nn.functional.conv2d(
        reshaped,
        weight_1d.view(1, 1, 1, -1),
        stride=(1, stride),
    )
    return conv_out.reshape(*arr.shape[:-1], -1)


def fw_arange(upper_bound, fw, device):
    return fw.arange(upper_bound) if fw is numpy else fw.arange(upper_bound, device=device)


def fw_empty(shape, fw, device):
    return fw.empty(shape) if fw is numpy else fw.empty(size=tuple(shape), device=device)
