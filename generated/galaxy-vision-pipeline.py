"""
Module implementing the FlowDIS algorithm from the paper:
"FlowDIS: Language-Guided Dichotomous Image Segmentation with Flow Matching"
available at http://arxiv.org/abs/2605.05077v1.

The mathematical idea behind this paper is to perform dichotomous image segmentation
using a flow matching approach guided by language. This is achieved by computing a
flow field between the input image and a semantic map, and then using this flow field
to guide the segmentation process.

Key hyperparameters:
- `alpha` (default: 0.5): controls the trade-off between the data term and the smoothness term
- `beta` (default: 0.1): controls the trade-off between the flow matching term and the segmentation term
- `num_iterations` (default: 10): number of iterations for the optimization process

Functions:
- `flow_dis`: computes the FlowDIS segmentation
- `galaxy_vision_pipeline`: applies the FlowDIS algorithm to the galaxy vision pipeline
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple

def flow_dis(image: torch.Tensor, semantic_map: torch.Tensor, alpha: float = 0.5, beta: float = 0.1, num_iterations: int = 10) -> torch.Tensor:
    """
    Computes the FlowDIS segmentation.

    Args:
    - image (torch.Tensor): input image
    - semantic_map (torch.Tensor): semantic map
    - alpha (float): controls the trade-off between the data term and the smoothness term (default: 0.5)
    - beta (float): controls the trade-off between the flow matching term and the segmentation term (default: 0.1)
    - num_iterations (int): number of iterations for the optimization process (default: 10)

    Returns:
    - segmentation (torch.Tensor): resulting segmentation
    """
    # Initialize the flow field
    flow_field = torch.zeros_like(image)

    # Compute the flow field using the Horn-Schunck method
    for _ in range(num_iterations):
        # Compute the gradient of the image
        grad_image = torch.gradient(image, dim=(1, 2))

        # Compute the gradient of the semantic map
        grad_semantic_map = torch.gradient(semantic_map, dim=(1, 2))

        # Compute the flow field update
        flow_field_update = -alpha * (grad_image[0] * flow_field[:, 0, :, :] + grad_image[1] * flow_field[:, 1, :, :]) - beta * (grad_semantic_map[0] * flow_field[:, 0, :, :] + grad_semantic_map[1] * flow_field[:, 1, :, :])

        # Update the flow field
        flow_field += flow_field_update

    # Compute the segmentation using the flow field
    segmentation = F.softmax(flow_field, dim=1)

    return segmentation

def galaxy_vision_pipeline(image: torch.Tensor, semantic_map: torch.Tensor) -> torch.Tensor:
    """
    Applies the FlowDIS algorithm to the galaxy vision pipeline.

    Args:
    - image (torch.Tensor): input image
    - semantic_map (torch.Tensor): semantic map

    Returns:
    - segmentation (torch.Tensor): resulting segmentation
    """
    # Compute the FlowDIS segmentation
    segmentation = flow_dis(image, semantic_map)

    return segmentation

if __name__ == "__main__":
    # Create a test image and semantic map
    image = torch.randn(1, 3, 256, 256)
    semantic_map = torch.randn(1, 3, 256, 256)

    # Apply the FlowDIS algorithm to the galaxy vision pipeline
    segmentation = galaxy_vision_pipeline(image, semantic_map)

    # Print the resulting segmentation
    print(segmentation.shape)