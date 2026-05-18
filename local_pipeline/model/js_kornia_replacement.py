import torch
import torch.nn.functional as F
import numpy as np

# def get_perspective_transform_torch(src, dst):
#     """
#     Replace kornia.geometry.transform.get_perspective_transform
#     Calculates perspective transformation matrix from 4 point correspondences.
    
#     Args:
#         src: [B, 4, 2] source points
#         dst: [B, 4, 2] destination points
    
#     Returns:
#         H: [B, 3, 3] perspective transformation matrices
#     """
#     batch_size = src.shape[0]
#     device = src.device
#     dtype = src.dtype
    
#     # Build the system of equations for each batch
#     # For perspective transform: dst = H @ src (in homogeneous coordinates)
#     # We need to solve for H using 4 point correspondences
    
#     H_list = []
#     # for b in range(batch_size): # batch_size=1 
#     src_b = src[0]  # [4, 2]
#     dst_b = dst[0]  # [4, 2]
    
#     # Build matrix A for the equation Ah = 0
#     A = torch.zeros((8, 9), device=device, dtype=dtype)
    
#     for i in range(4):
#         x, y = src_b[i]
#         u, v = dst_b[i]
        
#         A[2*i] = torch.tensor([x, y, 1, 0, 0, 0, -u*x, -u*y, -u], device=device, dtype=dtype)
#         A[2*i + 1] = torch.tensor([0, 0, 0, x, y, 1, -v*x, -v*y, -v], device=device, dtype=dtype)
    
#     # Solve using SVD
#     # try: # no try-catch for static graph
#     U, S, Vh = torch.linalg.svd()  # XXX linlag.inverse breaks on Jetson
#     h = Vh[-1].to(device)
#     # U, S, Vh = torch.linalg.svd(A)
#     # h = Vh[-1, :]  # Last row of V (or last column of V^H)
#     H_b = h.reshape(3, 3)
    
#     # Normalize so that H[2,2] = 1
#     H_b = H_b / H_b[2, 2]

#     # except:
#     #     # If SVD fails, return identity matrix
#     #     H_b = torch.eye(3, device=device, dtype=dtype)
    
#     H_list.append(H_b)
    
#     H = torch.stack(H_list, dim=0)
#     return H

def get_perspective_transform_torch(src, dst):
    """
    Replace kornia.geometry.transform.get_perspective_transform
    Calculates perspective transformation matrix from 4 point correspondences.
    
    Args:
        src: [B, 4, 2] source points
        dst: [B, 4, 2] destination points
    
    Returns:
        H: [B, 3, 3] perspective transformation matrices
    """
    batch_size = src.shape[0]
    device = src.device
    dtype = src.dtype
    
    # Build the system of equations for each batch
    # For perspective transform: dst = H @ src (in homogeneous coordinates)
    # We need to solve for H using 4 point correspondences
    
    H_list = []
    for b in range(batch_size):
        src_b = src[b]  # [4, 2]
        dst_b = dst[b]  # [4, 2]
        
        # Build matrix A for the equation Ah = 0
        A = torch.zeros((8, 9), device=device, dtype=dtype)
        
        for i in range(4):
            x, y = src_b[i]
            u, v = dst_b[i]
            
            A[2*i] = torch.stack([
                x, y, torch.tensor(1., device=device, dtype=dtype),
                torch.tensor(0., device=device, dtype=dtype),
                torch.tensor(0., device=device, dtype=dtype),
                torch.tensor(0., device=device, dtype=dtype),
                -u*x, -u*y, -u
            ])

            A[2*i + 1] = torch.stack([
                torch.tensor(0., device=device, dtype=dtype),
                torch.tensor(0., device=device, dtype=dtype),
                torch.tensor(0., device=device, dtype=dtype),
                x, y, torch.tensor(1., device=device, dtype=dtype),
                -v*x, -v*y, -v
            ])
            # A[2*i] = torch.tensor([x, y, 1, 0, 0, 0, -u*x, -u*y, -u], device=device, dtype=dtype)
            # A[2*i + 1] = torch.tensor([0, 0, 0, x, y, 1, -v*x, -v*y, -v], device=device, dtype=dtype)
            
        
        # Solve using SVD
        U, S, Vh = torch.linalg.svd(A)
        # U, S, Vh = torch.linalg.svd(A.cpu())
        h = Vh[-1].to(device)
        H_b = h.reshape(3, 3)
        
        # Normalize so that H[2,2] = 1
        H_b = H_b / H_b[2, 2]

        
        H_list.append(H_b)
    
    H = torch.stack(H_list, dim=0)
    return H

# def get_perspective_transform_torch(src, dst):
#     """
#     Replace kornia.geometry.transform.get_perspective_transform
#     Calculates perspective transformation matrix from 4 point correspondences.
    
#     Args:
#         src: [B, 4, 2] source points
#         dst: [B, 4, 2] destination points
    
#     Returns:
#         H: [B, 3, 3] perspective transformation matrices
#     """
#     batch_size = src.shape[0]
#     device = src.device
#     dtype = src.dtype
    
#     # Handle batch dimension properly
#     H_list = []
    
#     # for b in range(batch_size):
#     src_b = src[0]  # [4, 2]
#     dst_b = dst[0]  # [4, 2]
    
#     # Build matrix A for the equation Ah = 0
#     A = torch.zeros((8, 9), device=device, dtype=dtype)
    
#     for i in range(4):
#         x = src_b[i, 0]
#         y = src_b[i, 1]
#         u = dst_b[i, 0]
#         v = dst_b[i, 1]
        
#         # FIXED: Use stack instead of torch.tensor for traceability
#         A[2*i] = torch.stack([
#             x, y, torch.ones_like(x),
#             torch.zeros_like(x), torch.zeros_like(x), torch.zeros_like(x),
#             -u*x, -u*y, -u
#         ])
        
#         A[2*i + 1] = torch.stack([
#             torch.zeros_like(x), torch.zeros_like(x), torch.zeros_like(x),
#             x, y, torch.ones_like(x),
#             -v*x, -v*y, -v
#         ])
    
#     # FIXED: Don't move to CPU - keep on same device
#     # Solve using SVD (no .cpu() call)
#     U, S, Vh = torch.linalg.svd(A)
#     h = Vh[-1, :]  # Last row of Vh is the solution
#     H_b = h.reshape(3, 3)
    
#     # Normalize so that H[2,2] = 1
#     H_b = H_b / H_b[2, 2]
#     H_list.append(H_b)
    
#     H = torch.stack(H_list, dim=0)
#     return H


def crop_and_resize_torch(images, boxes, output_size):
    """
    Replace kornia.geometry.bbox.crop_and_resize
    Crops and resizes images based on bounding boxes.
    
    Args:
        images: [B, C, H, W] input images
        boxes: [B, 4, 2] bounding box coordinates in format [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
               or [B, 4] in format [x1, y1, x2, y2]
        output_size: (height, width) tuple for output size
    
    Returns:
        cropped: [B, C, output_height, output_width] cropped and resized images
    """
    B, C, H, W = images.shape
    output_h, output_w = output_size
    device = images.device
    dtype = images.dtype
    
    # Handle different box formats
    # if boxes.dim() == 3 and boxes.shape[1] == 4 and boxes.shape[2] == 2: # no if-else for static graph
    # Format: [B, 4, 2] - convert to [x_min, y_min, x_max, y_max]
    x_coords = boxes[:, :, 0]
    y_coords = boxes[:, :, 1]
    x_min = x_coords.min(dim=1)[0]
    y_min = y_coords.min(dim=1)[0]
    x_max = x_coords.max(dim=1)[0]
    y_max = y_coords.max(dim=1)[0]
    # else:
    #     # Format: [B, 4] as [x1, y1, x2, y2]
    #     x_min = boxes[:, 0]
    #     y_min = boxes[:, 1]
    #     x_max = boxes[:, 2]
    #     y_max = boxes[:, 3]
    
    # Normalize coordinates to [-1, 1] for grid_sample
    # grid_sample expects coordinates in range [-1, 1]
    x_min_norm = (2.0 * x_min / (W - 1)) - 1.0
    x_max_norm = (2.0 * x_max / (W - 1)) - 1.0
    y_min_norm = (2.0 * y_min / (H - 1)) - 1.0
    y_max_norm = (2.0 * y_max / (H - 1)) - 1.0
    
    # Create sampling grid
    # grid_sample expects grid in shape [B, H, W, 2] with values in [-1, 1]
    y_grid = torch.linspace(0, 1, output_h, device=device, dtype=dtype)
    x_grid = torch.linspace(0, 1, output_w, device=device, dtype=dtype)
    
    y_grid, x_grid = torch.meshgrid(y_grid, x_grid, indexing='ij')
    
    # Expand for batch
    y_grid = y_grid.unsqueeze(0).expand(B, -1, -1)  # [B, H, W]
    x_grid = x_grid.unsqueeze(0).expand(B, -1, -1)  # [B, H, W]
    
    # Map from [0, 1] to box coordinates in normalized space [-1, 1]
    x_min_norm = x_min_norm.view(B, 1, 1)
    x_max_norm = x_max_norm.view(B, 1, 1)
    y_min_norm = y_min_norm.view(B, 1, 1)
    y_max_norm = y_max_norm.view(B, 1, 1)
    
    x_sample = x_min_norm + (x_max_norm - x_min_norm) * x_grid
    y_sample = y_min_norm + (y_max_norm - y_min_norm) * y_grid
    
    # Stack to create grid [B, H, W, 2]
    grid = torch.stack([x_sample, y_sample], dim=-1)
    
    # Use grid_sample to crop and resize
    # padding_mode='border' handles out-of-boundary cases similar to kornia
    cropped = F.grid_sample(images, grid, mode='bilinear', 
                           padding_mode='border', align_corners=True)
    
    return cropped


def bbox_generator_torch(x_start, y_start, width, height):
    """
    Generate bounding boxes in the format compatible with crop_and_resize_torch
    
    Args:
        x_start: [B] x coordinates of top-left corner
        y_start: [B] y coordinates of top-left corner
        width: [B] width of boxes
        height: [B] height of boxes
    
    Returns:
        boxes: [B, 4, 2] bounding box coordinates [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    """
    B = x_start.shape[0]
    device = x_start.device
    
    boxes = torch.zeros((B, 4, 2), device=device, dtype=x_start.dtype)
    
    # Top-left
    boxes[:, 0, 0] = x_start
    boxes[:, 0, 1] = y_start
    
    # Top-right
    boxes[:, 1, 0] = x_start + width
    boxes[:, 1, 1] = y_start
    
    # Bottom-right
    boxes[:, 2, 0] = x_start + width
    boxes[:, 2, 1] = y_start + height
    
    # Bottom-left
    boxes[:, 3, 0] = x_start
    boxes[:, 3, 1] = y_start + height
    
    return boxes


def warp_perspective_torch(src, M, dsize, mode='bilinear', padding_mode='zeros'):
    """
    Replace kornia.geometry.transform.warp_perspective
    Applies a perspective transformation to an image.
    
    Args:
        src: [B, C, H, W] input images
        M: [B, 3, 3] perspective transformation matrices
        dsize: (height, width) tuple for output size
        mode: interpolation mode ('bilinear' or 'nearest')
        padding_mode: padding mode for grid_sample ('zeros', 'border', 'reflection')
    
    Returns:
        warped: [B, C, output_height, output_width] warped images
    """
    B, C, H_in, W_in = src.shape
    H_out, W_out = dsize
    device = src.device
    dtype = src.dtype
    
    # Create destination grid (where we want to sample from)
    # Grid coordinates in output image space
    y_grid = torch.linspace(0, H_out - 1, H_out, device=device, dtype=dtype)
    x_grid = torch.linspace(0, W_out - 1, W_out, device=device, dtype=dtype)
    y_grid, x_grid = torch.meshgrid(y_grid, x_grid, indexing='ij')
    
    # Create homogeneous coordinates [x, y, 1] for each pixel in output
    ones = torch.ones_like(x_grid)
    grid = torch.stack([x_grid, y_grid, ones], dim=0)  # [3, H_out, W_out]
    grid = grid.view(3, -1)  # [3, H_out*W_out]
    
    # Expand for batch
    grid = grid.unsqueeze(0).expand(B, -1, -1)  # [B, 3, H_out*W_out]
    
    # Apply inverse perspective transformation
    # We need to find where each output pixel comes from in the input
    # src_coords = M^(-1) @ dst_coords
    try:
        M_inv = torch.inverse(M)  # [B, 3, 3]
    except:
        # If matrix is singular, use pseudo-inverse
        M_inv = torch.linalg.pinv(M)
    
    # Transform coordinates
    src_coords = torch.bmm(M_inv, grid)  # [B, 3, H_out*W_out]
    
    # Convert from homogeneous to Cartesian coordinates
    # Avoid division by zero
    z = src_coords[:, 2:3, :]  # [B, 1, H_out*W_out]
    z = torch.where(torch.abs(z) < 1e-8, torch.ones_like(z), z)
    
    src_coords = src_coords[:, :2, :] / z  # [B, 2, H_out*W_out]
    
    # Reshape to grid format
    src_x = src_coords[:, 0, :].view(B, H_out, W_out)  # [B, H_out, W_out]
    src_y = src_coords[:, 1, :].view(B, H_out, W_out)  # [B, H_out, W_out]
    
    # Normalize coordinates to [-1, 1] for grid_sample
    # grid_sample expects (x, y) coordinates normalized to [-1, 1]
    src_x_norm = 2.0 * src_x / (W_in - 1) - 1.0
    src_y_norm = 2.0 * src_y / (H_in - 1) - 1.0
    
    # Stack to create sampling grid [B, H_out, W_out, 2]
    grid_sample = torch.stack([src_x_norm, src_y_norm], dim=-1)
    
    # Sample from source image
    warped = F.grid_sample(
        src, 
        grid_sample, 
        mode=mode, 
        padding_mode=padding_mode, 
        align_corners=True
    )
    
    return warped

def normalize_points(pts):
    mean = pts.mean(dim=0)
    std = pts.std(dim=0).mean()
    pts_norm = (pts - mean) / std

    T = torch.tensor([
        [1/std, 0, -mean[0]/std],
        [0, 1/std, -mean[1]/std],
        [0, 0, 1]
    ], device=pts.device, dtype=pts.dtype)

    return pts_norm, T
