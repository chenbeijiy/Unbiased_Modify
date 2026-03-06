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
    # Get camera center in world coordinates
    camera_center = viewpoint_cam.camera_center  # [3]
    
    # Reconstruct 3D points from depth map
    points_3d = depths_to_points(viewpoint_cam, surf_depth)  # [H, W, 3]
    
    # Compute view direction: from surface points to camera center
    # Broadcast camera_center [3] to [H, W, 3]
    H, W = points_3d.shape[:2]
    camera_center_expanded = camera_center.view(1, 1, 3).expand(H, W, -1)  # [H, W, 3]
    view_dirs = camera_center_expanded - points_3d  # [H, W, 3]
    
    # Normalize view direction vectors
    view_dirs_norm = torch.norm(view_dirs, dim=-1, keepdim=True)
    view_dirs = view_dirs / (view_dirs_norm + 1e-8)  # [H, W, 3]
    
    return view_dirs


def compute_view_normal_angle(view_dirs, surf_normal):
    """
    Compute cosine of angle between view direction and surface normal.
    
    Args:
        view_dirs: View direction vectors [H, W, 3] (normalized)
        surf_normal: Surface normal map [3, H, W] or [H, W, 3]
    
    Returns:
        cos_theta: Cosine of angle between view direction and normal [H, W]
    """
    # Ensure surf_normal is [H, W, 3]
    if surf_normal.dim() == 3 and surf_normal.shape[0] == 3:
        # [3, H, W] -> [H, W, 3]
        surf_normal = surf_normal.permute(1, 2, 0)
    
    # Normalize surface normals
    surf_normal_norm = torch.norm(surf_normal, dim=-1, keepdim=True)
    surf_normal_normalized = surf_normal / (surf_normal_norm + 1e-8)
    
    # Compute cosine of angle: dot(view_dir, normal)
    cos_theta = (view_dirs * surf_normal_normalized).sum(dim=-1)  # [H, W]
    
    # Clamp to [-1, 1] for numerical stability
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    
    return cos_theta


def compute_depth_gradient(surf_depth):
    """
    Compute squared depth gradient magnitude.
    
    Args:
        surf_depth: Surface depth map [1, H, W]
    
    Returns:
        depth_grad_sq: Squared depth gradient magnitude [1, H, W]
    """
    # Ensure surf_depth is [1, H, W]
    if surf_depth.dim() == 2:
        surf_depth = surf_depth.unsqueeze(0)
    
    # Compute gradients using Sobel operator or finite differences
    # Use central differences for better accuracy
    # Gradient in x direction
    depth_grad_x = surf_depth[:, :, 1:] - surf_depth[:, :, :-1]  # [1, H, W-1]
    # Pad to match original size
    depth_grad_x = F.pad(depth_grad_x, (0, 1), mode='replicate')  # [1, H, W]
    
    # Gradient in y direction
    depth_grad_y = surf_depth[:, 1:, :] - surf_depth[:, :-1, :]  # [1, H-1, W]
    # Pad to match original size
    depth_grad_y = F.pad(depth_grad_y, (0, 0, 0, 1), mode='replicate')  # [1, H, W]
    
    # Compute squared gradient magnitude
    depth_grad_sq = depth_grad_x ** 2 + depth_grad_y ** 2  # [1, H, W]
    
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
    Compute view-dependent depth constraint loss.
    
    This loss enforces depth smoothness with view-dependent weighting:
    - Stronger constraint in front-facing views (view-normal angle small)
    - Weaker constraint in side-facing views (view-normal angle large)
    
    Args:
        render_pkg: Render package containing:
            - 'surf_depth': Surface depth map [1, H, W]
            - 'surf_normal': Surface normal map [3, H, W]
        viewpoint_cam: Camera object with camera_center attribute
        lambda_weight: Overall weight for the loss
        lambda_view_weight: Weight parameter for view-dependent weighting (default: 2.0)
        mask_background: Whether to mask background pixels
        background_threshold_factor: Factor for background threshold (default: 0.95)
    
    Returns:
        loss: Scalar tensor representing the view-dependent depth constraint loss
    """
    # Extract depth and normal from render package
    surf_depth = render_pkg['surf_depth']  # [1, H, W]
    surf_normal = render_pkg['surf_normal']  # [3, H, W]
    
    H, W = surf_depth.shape[-2:]
    
    # Step 1: Compute view direction vectors
    view_dirs = compute_view_direction(viewpoint_cam, surf_depth)  # [H, W, 3]
    
    # Step 2: Compute cosine of angle between view direction and surface normal
    cos_theta = compute_view_normal_angle(view_dirs, surf_normal)  # [H, W]
    
    # Step 3: Compute view-dependent weight
    # Linear weight: more stable
    view_weight_linear = 0.1 + 0.9 * (cos_theta + 1.0) / 2.0  # [H, W], range: [0.1, 1.0]
    
    # Exp-based weight: original formula
    lambda_view_weight_reduced = lambda_view_weight / 2.0  # Reduce for stability
    view_weight_exp = torch.exp(-lambda_view_weight_reduced * (1.0 - cos_theta))  # [H, W]
    
    # Blend both weights: 70% linear + 30% exp
    view_weight = 0.7 * view_weight_linear + 0.3 * view_weight_exp  # [H, W]
    
    # Step 4: Compute depth gradient
    depth_grad_sq = compute_depth_gradient(surf_depth)  # [1, H, W]
    
    # Reshape view_weight to match depth_grad_sq
    view_weight = view_weight.unsqueeze(0)  # [1, H, W]
    
    # Step 5: Create mask
    mask = torch.ones_like(depth_grad_sq)
    
    # Mask background (very far depths)
    if mask_background:
        max_depth = surf_depth.max()
        background_threshold = max_depth * background_threshold_factor
        mask = mask * (surf_depth < background_threshold).float()
    
    # Step 6: Apply view-dependent weight to depth gradient
    weighted_depth_grad = view_weight * depth_grad_sq * mask  # [1, H, W]
    
    # Step 7: Compute mean loss
    loss = weighted_depth_grad.mean()
    
    return lambda_weight * loss
