#
# View-Dependent Depth Constraint Loss Implementation
# Innovation Point 3: View-Dependent Depth Constraint
#

import torch
import torch.nn.functional as F
from utils.point_utils import depths_to_points


def compute_view_direction(viewpoint_cam, surf_depth):
    """
    Compute view direction vectors from surface points to camera center.
    
    Args:
        viewpoint_cam: Camera object with camera_center attribute
        surf_depth: Surface depth map [1, H, W]
    
    Returns:
        view_dirs: View direction vectors [H, W, 3] (normalized, pointing from surface to camera)
    """
    # depths_to_points returns [H*W, 3], need reshape to [H, W, 3]
    points_flat = depths_to_points(viewpoint_cam, surf_depth)  # [H*W, 3]
    H, W = surf_depth.shape[-2:]
    points_3d = points_flat.reshape(H, W, 3)  # [H, W, 3]
    
    # Camera center [3], broadcast to [H, W, 3]
    camera_center = viewpoint_cam.camera_center  # [3]
    view_dirs = camera_center.view(1, 1, 3).expand(H, W, 3) - points_3d  # [H, W, 3]
    
    # Normalize
    view_dirs_norm = torch.norm(view_dirs, dim=-1, keepdim=True).clamp(min=1e-8)
    view_dirs = view_dirs / view_dirs_norm  # [H, W, 3]
    
    return view_dirs


def compute_view_normal_angle(view_dirs, surf_normal):
    """
    Compute cosine of angle between view direction and surface normal.
    
    Args:
        view_dirs: [H, W, 3]
        surf_normal: [3, H, W]
    
    Returns:
        cos_theta: [H, W]
    """
    # [3, H, W] -> [H, W, 3]
    surf_normal_hw = surf_normal.permute(1, 2, 0)  # [H, W, 3]
    surf_normal_norm = torch.norm(surf_normal_hw, dim=-1, keepdim=True).clamp(min=1e-8)
    surf_normal_hw = surf_normal_hw / surf_normal_norm
    
    cos_theta = (view_dirs * surf_normal_hw).sum(dim=-1).clamp(-1.0, 1.0)
    return cos_theta


def compute_depth_gradient(surf_depth):
    """
    Compute squared depth gradient magnitude.
    
    Args:
        surf_depth: [1, H, W]
    
    Returns:
        depth_grad_sq: [1, H, W]
    """
    if surf_depth.dim() == 2:
        surf_depth = surf_depth.unsqueeze(0)
    
    grad_x = surf_depth[:, :, 1:] - surf_depth[:, :, :-1]  # [1, H, W-1]
    grad_x = F.pad(grad_x, (0, 1), mode='replicate')  # [1, H, W]
    
    grad_y = surf_depth[:, 1:, :] - surf_depth[:, :-1, :]  # [1, H-1, W]
    grad_y = F.pad(grad_y, (0, 0, 0, 1), mode='replicate')  # [1, H, W]
    
    depth_grad_sq = grad_x ** 2 + grad_y ** 2
    return depth_grad_sq


def view_dependent_depth_constraint_loss(
    render_pkg,
    viewpoint_cam,
    lambda_weight=1.0,
    lambda_view_weight=2.0,
    mask_background=True,
    background_threshold_factor=0.95
):
    """
    View-dependent depth constraint loss.
    
    Args:
        render_pkg: dict with 'surf_depth' [1,H,W], 'surf_normal' [3,H,W]
        viewpoint_cam: camera with camera_center
        lambda_weight: loss weight
        lambda_view_weight: view weight param
        mask_background: whether to mask background
        background_threshold_factor: background threshold
    """
    surf_depth = render_pkg['surf_depth']  # [1, H, W]
    surf_normal = render_pkg['surf_normal']  # [3, H, W]
    
    view_dirs = compute_view_direction(viewpoint_cam, surf_depth)  # [H, W, 3]
    cos_theta = compute_view_normal_angle(view_dirs, surf_normal)  # [H, W]
    
    view_weight_linear = 0.1 + 0.9 * (cos_theta + 1.0) / 2.0
    lambda_reduced = lambda_view_weight / 2.0
    view_weight_exp = torch.exp(-lambda_reduced * (1.0 - cos_theta))
    view_weight = 0.7 * view_weight_linear + 0.3 * view_weight_exp  # [H, W]
    
    depth_grad_sq = compute_depth_gradient(surf_depth)  # [1, H, W]
    view_weight = view_weight.unsqueeze(0)  # [1, H, W]
    
    mask = torch.ones_like(depth_grad_sq)
    if mask_background:
        thresh = surf_depth.max() * background_threshold_factor
        mask = mask * (surf_depth < thresh).float()
    
    weighted_grad = view_weight * depth_grad_sq * mask
    loss = weighted_grad.mean()
    
    return lambda_weight * loss
