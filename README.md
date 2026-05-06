# Galaxy Vision Pipeline - Repository Context

## Overview
This repository contains the core software for the Samsung Galaxy Vision AI pipeline. It handles direct sensor interfacing, image signal processing (ISP), and on-device AI enhancements like super-resolution and nightography denoising.

## Tech Stack & Architecture
- **Language**: Python (simulating native C++ pipelines)
- **AI Models**: Spatial CNNs, currently evaluating Vision Transformers (ViTs).
- **Constraints**: 
  - Must run inference in under 50ms for video.
  - Strict memory limitations for edge hardware.
  - High battery consumption is a major concern.

## Current Goals & Challenges
1. **Migrate to Transformers**: We want to replace our aging CNN-based super-resolution with modern Vision Transformers (ViT), but they are currently too slow and memory-heavy for mobile.
2. **Improve Low-Light Denoising**: Seeking more efficient attention mechanisms that don't scale quadratically with image resolution.
3. **Hardware Acceleration**: Looking for ways to better utilize the Exynos NPU for vision tasks.
