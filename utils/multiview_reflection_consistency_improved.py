#
# Multi-View Reflection Consistency Loss Implementation
# Innovation Point 2: Multi-View Reflection Consistency Constraint
#

import torch
import torch.nn.functional as F


def compute_luminance(rgb_image):
    """
    Compute luminance from RGB image.
    
    Args:
        rgb_image: Tensor of shape [C, H, W] or [1, C, H, W], where C=3 for RGB
    
    Returns:
        luminance: Tensor of shape [1, H, W] or [H, W]
    """
    if rgb_image.dim() == 4:
        # [B, C, H, W] -> [B, 1, H, W]
        luminance = rgb_image.mean(dim=1, keepdim=True)
    else:
        # [C, H, W] -> [1, H, W]
        luminance = rgb_image.mean(dim=0, keepdim=True)
    return luminance


def compute_rgb_variance(rgb_image):
    """
    Compute RGB variance for reflection detection.
    
    Args:
        rgb_image: Tensor of shape [C, H, W] or [1, C, H, W], where C=3 for RGB
    
    Returns:
        variance: Tensor of shape [1, H, W] or [H, W]
    """
    if rgb_image.dim() == 4:
        # [B, C, H, W]
        rgb_mean = rgb_image.mean(dim=1, keepdim=True)  # [B, 1, H, W]
        rgb_variance = ((rgb_image - rgb_mean) ** 2).mean(dim=1, keepdim=True)  # [B, 1, H, W]
    else:
        # [C, H, W]
        rgb_mean = rgb_image.mean(dim=0, keepdim=True)  # [1, H, W]
        rgb_variance = ((rgb_image - rgb_mean) ** 2).mean(dim=0, keepdim=True)  # [1, H, W]
    return rgb_variance


def compute_reflection_weight(rgb_image, sigmoid_scale=10.0):
    """
    Compute reflection weight based on RGB luminance and variance.
    
    High luminance + high variance indicates specular highlights.
    
    Args:
        rgb_image: Tensor of shape [C, H, W] or [1, C, H, W], where C=3 for RGB
        sigmoid_scale: Scale factor for sigmoid normalization
    
    Returns:
        reflection_weight: Tensor of shape [1, H, W] or [H, W], values in [0, 1]
    """
    luminance = compute_luminance(rgb_image)
    rgb_variance = compute_rgb_variance(rgb_image)
    
    # Specular strength: high luminance * high variance (highlights)
    specular_strength = luminance * rgb_variance
    
    # Normalize to [0, 1] using sigmoid
    reflection_weight = torch.sigmoid(sigmoid_scale * specular_strength)
    
    return reflection_weight


def multiview_reflection_consistency_loss_improved(
    render_pkgs,
    viewpoint_cameras,
    lambda_weight=1.0,
    mask_background=True,
    use_highlight_mask=True,
    highlight_threshold=0.5,
    resolution_scale=1.0,
    sigmoid_scale=10.0
):
    """
    Improved multi-view reflection consistency loss.
    
    This loss enforces depth consistency across multiple viewpoints,
    with higher weight in reflection/specular regions.
    
    Args:
        render_pkgs: List of render packages, each containing:
            - 'render': RGB image [3, H, W]
            - 'surf_depth': Surface depth map [1, H, W]
        viewpoint_cameras: List of viewpoint cameras (for potential use)
        lambda_weight: Weight for the loss
        mask_background: Whether to mask background pixels
        use_highlight_mask: Whether to only compute loss in highlight regions
        highlight_threshold: Threshold for highlight mask
        resolution_scale: Scale factor for resolution (for efficiency)
        sigmoid_scale: Scale factor for sigmoid in reflection weight
    
    Returns:
        loss: Scalar tensor representing the multi-view reflection consistency loss
    """
    if len(render_pkgs) < 2:
        # Need at least 2 viewpoints for multi-view consistency
        if len(render_pkgs) > 0:
            return torch.tensor(0.0, device=render_pkgs[0]['render'].device, requires_grad=True)
        else:
            # Fallback: return zero tensor on CPU (should not happen in practice)
            return torch.tensor(0.0, requires_grad=True)
    
    # Initialize total_loss as tensor to avoid type mismatch
    total_loss = torch.tensor(0.0, device=render_pkgs[0]['render'].device, requires_grad=True)
    num_pairs = 0
    
    # Extract depths and RGB images from all render packages
    depths = []
    rgb_images = []
    
    for render_pkg in render_pkgs:
        rgb_image = render_pkg['render']  # [3, H, W]
        surf_depth = render_pkg['surf_depth']  # [1, H, W]
        
        # Apply resolution scaling if needed
        if resolution_scale < 1.0:
            H, W = rgb_image.shape[-2:]
            new_H, new_W = int(H * resolution_scale), int(W * resolution_scale)
            rgb_image = F.interpolate(rgb_image.unsqueeze(0), size=(new_H, new_W), mode='bilinear', align_corners=False).squeeze(0)
            surf_depth = F.interpolate(surf_depth.unsqueeze(0), size=(new_H, new_W), mode='bilinear', align_corners=False).squeeze(0)
        
        rgb_images.append(rgb_image)
        depths.append(surf_depth)
    
    # Compute reflection weights for each view
    reflection_weights = []
    for rgb_image in rgb_images:
        reflection_weight = compute_reflection_weight(rgb_image, sigmoid_scale=sigmoid_scale)
        reflection_weights.append(reflection_weight)
    
    # Unify spatial size to avoid shape mismatch when cameras have different resolutions
    target_H, target_W = depths[0].shape[-2], depths[0].shape[-1]
    for idx in range(len(depths)):
        d = depths[idx]  # [1, H, W]
        if d.shape[-2] != target_H or d.shape[-1] != target_W:
            depths[idx] = F.interpolate(d.unsqueeze(0), size=(target_H, target_W), mode='bilinear', align_corners=False).squeeze(0)
        r = reflection_weights[idx]  # [1, H, W]
        if r.shape[-2] != target_H or r.shape[-1] != target_W:
            reflection_weights[idx] = F.interpolate(r.unsqueeze(0), size=(target_H, target_W), mode='bilinear', align_corners=False).squeeze(0)
    
    # Compute pairwise depth consistency loss (each pair (i,j) computed once)
    for i in range(len(render_pkgs)):
        for j in range(i + 1, len(render_pkgs)):
            depth_i = depths[i]  # [1, H, W]
            depth_j = depths[j]  # [1, H, W]
            
            # Compute depth difference
            depth_diff = depth_i - depth_j  # [1, H, W]
            depth_diff_sq = depth_diff ** 2  # [1, H, W]
            
            # Compute reflection weight (use minimum to be conservative)
            reflection_weight_i = reflection_weights[i]  # [1, H, W]
            reflection_weight_j = reflection_weights[j]  # [1, H, W]
            reflection_weight = torch.min(reflection_weight_i, reflection_weight_j)  # [1, H, W]
            
            # Create mask
            mask = torch.ones_like(depth_diff)
            
            # Mask background (very far depths)
            if mask_background:
                # Assume background has very large depth values
                max_depth = torch.max(depth_i.max(), depth_j.max())
                background_threshold = max_depth * 0.95  # 95% of max depth
                mask = mask * (depth_i < background_threshold).float() * (depth_j < background_threshold).float()
            
            # Apply highlight mask if enabled
            if use_highlight_mask:
                highlight_mask = (reflection_weight > highlight_threshold).float()
                mask = mask * highlight_mask
            
            # Compute weighted loss
            weighted_loss = reflection_weight * depth_diff_sq * mask
            # Use = instead of += to avoid in-place op on leaf tensor (which requires grad)
            total_loss = total_loss + weighted_loss.mean()
            num_pairs += 1
    
    # Average over all pairs
    if num_pairs > 0:
        total_loss = total_loss / num_pairs
    
    return lambda_weight * total_loss
