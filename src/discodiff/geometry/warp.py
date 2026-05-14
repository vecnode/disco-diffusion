import os

import torch, torchvision
from . import py3d_tools as p3d
from PIL import Image
import numpy as np
import math
import cv2


def _save_3d_debug(debug_dir: str, frame_num: int, name: str, data) -> None:
    """Save a debug image (numpy array or PIL Image) to debug_dir/frame_NNNN_<name>.png."""
    if debug_dir is None:
        return
    os.makedirs(debug_dir, exist_ok=True)
    path = os.path.join(debug_dir, f"frame_{frame_num:04d}_{name}.png")
    if isinstance(data, Image.Image):
        data.save(path)
        return
    arr = np.nan_to_num(np.array(data, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    # Normalize to [0, 255]
    lo, hi = arr.min(), arr.max()
    if hi > lo:
        arr = (arr - lo) / (hi - lo)
    else:
        arr = np.zeros_like(arr)
    arr = (arr * 255).clip(0, 255).astype(np.uint8)
    if arr.ndim == 2:
        cv2.imwrite(path, arr)
    else:
        # Assume HxWx2 flow field — save as HSV colour wheel visualization
        h, w = arr.shape[:2]
        flow = np.nan_to_num(np.array(data, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv = np.zeros((h, w, 3), dtype=np.uint8)
        hsv[..., 0] = (ang * 180 / np.pi / 2).astype(np.uint8)
        hsv[..., 1] = 255
        mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        hsv[..., 2] = mag_norm.astype(np.uint8)
        cv2.imwrite(path, cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR))

MAX_ADABINS_AREA = 500000
MIN_ADABINS_AREA = 448*448

_DEPTH_EMA_CACHE = {}


def _robust_normalize_depth(depth_map: np.ndarray) -> np.ndarray:
    depth_arr = np.nan_to_num(np.asarray(depth_map, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if not np.isfinite(depth_arr).any():
        return np.zeros_like(depth_arr, dtype=np.float32)
    lo, hi = np.percentile(depth_arr, [2.0, 98.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-6:
        return np.zeros_like(depth_arr, dtype=np.float32)
    depth_arr = np.clip(depth_arr, lo, hi)
    return (depth_arr - lo) / (hi - lo)


def _smooth_depth_temporally(depth_map: np.ndarray, frame_num: int, cache_key: str) -> np.ndarray:
    depth_arr = np.asarray(depth_map, dtype=np.float32)
    state = _DEPTH_EMA_CACHE.get(cache_key)
    if frame_num <= 0 or state is None or state.get("value") is None or state.get("frame") is None:
        _DEPTH_EMA_CACHE[cache_key] = {"frame": frame_num, "value": depth_arr.copy()}
        return depth_arr

    previous_frame = int(state["frame"])
    previous_depth = np.asarray(state["value"], dtype=np.float32)
    if previous_frame != frame_num - 1 or previous_depth.shape != depth_arr.shape:
        _DEPTH_EMA_CACHE[cache_key] = {"frame": frame_num, "value": depth_arr.copy()}
        return depth_arr

    warmup = min(1.0, frame_num / 24.0)
    alpha = 0.12 + 0.28 * warmup
    smoothed = alpha * depth_arr + (1.0 - alpha) * previous_depth
    _DEPTH_EMA_CACHE[cache_key] = {"frame": frame_num, "value": smoothed.copy()}
    return smoothed

@torch.no_grad()
def transform_image_3d(img_filepath, depth_backend, device, rot_mat=torch.eye(3).unsqueeze(0), translate=(0.,0.,-0.04), near=2000, far=20000, fov_deg=60, padding_mode='border', sampling_mode='bicubic', spherical=False, debug_dir=None, frame_num=0):
    img_pil = Image.open(open(img_filepath, 'rb')).convert('RGB')
    w, h = img_pil.size
    image_tensor = torchvision.transforms.functional.to_tensor(img_pil).to(device)

    if depth_backend is None:
        raise RuntimeError("Depth backend is not initialized")

    image_pil_area = w * h
    if image_pil_area > MAX_ADABINS_AREA:
        scale = math.sqrt(MAX_ADABINS_AREA) / math.sqrt(image_pil_area)
        depth_input = img_pil.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    elif image_pil_area < MIN_ADABINS_AREA:
        scale = math.sqrt(MIN_ADABINS_AREA) / math.sqrt(image_pil_area)
        depth_input = img_pil.resize((int(w * scale), int(h * scale)), Image.BICUBIC)
    else:
        depth_input = img_pil

    try:
        depth_np = depth_backend.predict_depth(depth_input, (w, h))
    except Exception as e:
        raise RuntimeError(f"Depth prediction failed: {e}") from e

    torch.cuda.empty_cache()

    depth_map = _robust_normalize_depth(depth_np)
    depth_contrast = float(getattr(depth_backend, "depth_contrast", 1.0))
    if abs(depth_contrast - 1.0) > 1e-6:
        depth_map = np.clip(0.5 + ((depth_map - 0.5) * depth_contrast), 0.0, 1.0)
    _save_3d_debug(debug_dir, frame_num, 'depth_backend', depth_map)

    depth_map = _smooth_depth_temporally(depth_map, frame_num, str(debug_dir or "default"))
    depth_map = 0.5 + (2.0 * depth_map)
    _save_3d_debug(debug_dir, frame_num, 'depth_blended', depth_map)

    depth_map = np.expand_dims(depth_map, axis=0)
    depth_tensor = torch.from_numpy(depth_map).squeeze().to(device)
    depth_tensor = torch.nan_to_num(depth_tensor, nan=0.0, posinf=0.0, neginf=0.0)

    pixel_aspect = 1.0 # really.. the aspect of an individual pixel! (so usually 1.0)
    persp_cam_old = p3d.FoVPerspectiveCameras(near, far, pixel_aspect, fov=fov_deg, degrees=True, device=device)
    persp_cam_new = p3d.FoVPerspectiveCameras(near, far, pixel_aspect, fov=fov_deg, degrees=True, R=rot_mat, T=torch.tensor([translate]), device=device)

    # range of [-1,1] is important to torch grid_sample's padding handling
    y,x = torch.meshgrid(torch.linspace(-1.,1.,h,dtype=torch.float32,device=device),torch.linspace(-1.,1.,w,dtype=torch.float32,device=device))
    z = torch.as_tensor(depth_tensor, dtype=torch.float32, device=device)
    xyz_old_world = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)

    # Transform the points using pytorch3d. With current functionality, this is overkill and prevents it from working on Windows.
    # If you want it to run on Windows (without pytorch3d), then the transforms (and/or perspective if that's separate) can be done pretty easily without it.
    xyz_old_cam_xy = persp_cam_old.get_full_projection_transform().transform_points(xyz_old_world)[:,0:2]
    xyz_new_cam_xy = persp_cam_new.get_full_projection_transform().transform_points(xyz_old_world)[:,0:2]

    offset_xy = torch.nan_to_num(xyz_new_cam_xy - xyz_old_cam_xy, nan=0.0, posinf=0.0, neginf=0.0)
    _save_3d_debug(debug_dir, frame_num, 'flow_field', offset_xy.reshape(h, w, 2).cpu().numpy())
    # affine_grid theta param expects a batch of 2D mats. Each is 2x3 to do rotation+translation.
    identity_2d_batch = torch.tensor([[1.,0.,0.],[0.,1.,0.]], device=device).unsqueeze(0)
    # coords_2d will have shape (N,H,W,2).. which is also what grid_sample needs.
    coords_2d = torch.nn.functional.affine_grid(identity_2d_batch, [1,1,h,w], align_corners=False)
    offset_coords_2d = coords_2d - torch.reshape(offset_xy, (h,w,2)).unsqueeze(0)

    if spherical:
        spherical_grid = get_spherical_projection(h, w, torch.tensor([0,0], device=device), -0.4,device=device)#align_corners=False
        stage_image = torch.nn.functional.grid_sample(image_tensor.add(1/512 - 0.0001).unsqueeze(0), offset_coords_2d, mode=sampling_mode, padding_mode=padding_mode, align_corners=True)
        new_image = torch.nn.functional.grid_sample(stage_image, spherical_grid,align_corners=True) #, mode=sampling_mode, padding_mode=padding_mode, align_corners=False)
    else:
        new_image = torch.nn.functional.grid_sample(image_tensor.add(1/512 - 0.0001).unsqueeze(0), offset_coords_2d, mode=sampling_mode, padding_mode=padding_mode, align_corners=False)

    new_image = torch.nan_to_num(new_image, nan=0.0, posinf=0.0, neginf=0.0)

    img_pil = torchvision.transforms.ToPILImage()(new_image.squeeze().clamp(0,1.))
    _save_3d_debug(debug_dir, frame_num, 'warped', img_pil)
    _save_3d_debug(debug_dir, frame_num, 'source', Image.open(open(img_filepath, 'rb')).convert('RGB'))

    torch.cuda.empty_cache()

    return img_pil

def get_spherical_projection(H, W, center, magnitude,device):  
    xx, yy = torch.linspace(-1, 1, W,dtype=torch.float32,device=device), torch.linspace(-1, 1, H,dtype=torch.float32,device=device)  
    gridy, gridx  = torch.meshgrid(yy, xx)
    grid = torch.stack([gridx, gridy], dim=-1)  
    d = center - grid
    d_sum = torch.sqrt((d**2).sum(axis=-1))
    grid += d * d_sum.unsqueeze(-1) * magnitude 
    return grid.unsqueeze(0)

