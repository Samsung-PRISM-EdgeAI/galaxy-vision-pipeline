import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import math

# --- Module-level Docstring ---
"""
FlowDIS: Language-Guided Dichotomous Image Segmentation with Flow Matching

Source Paper: https://www.semanticscholar.org/paper/2605.05077v1

This module implements the core algorithm of FlowDIS, a novel method for
dichotomous (binary) image segmentation guided by natural language prompts.
The method leverages Flow Matching, a technique that learns a continuous-time
vector field to transform a simple prior distribution (e.g., Gaussian noise)
into the target data distribution (segmentation masks).

Mathematical Idea:
The core idea is to model the evolution of a noisy segmentation mask `x_t`
from `t=0` (pure noise) to `t=1` (the target segmentation mask) as a
continuous ordinary differential equation (ODE): `dx_t / dt = v_t(x_t, t, condition)`.
Here, `v_t` is a vector field (velocity) predicted by a neural network (typically a U-Net).
The `condition` includes both a language embedding (derived from a text prompt)
and features extracted from the input image.

During training (not implemented here, assumed to be pre-trained), the model
`f_theta(x_t, t, condition)` is optimized to predict the true velocity