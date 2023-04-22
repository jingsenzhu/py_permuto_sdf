# Permuto SDF: Pytorch Extension
An unofficial Pytorch extension for occupancy grids and volume rendering in Permuto SDF (CVPR 2023). Original repository: [https://github.com/RaduAlexandru/permuto_sdf](https://github.com/RaduAlexandru/permuto_sdf).

## Install

```shell
cd py_permuto_sdf
pip install -e .
```

Requires Pytorch and CUDA. Tested on Pytorch version `1.12.1` + CUDA `11.3` on both Linux and Windows 10.

## Usage

```python
from py_permuto_sdf import OccupancyGrid, Sphere, RaySampler, VolumeRendering
```

### Changes to original

Most functions and APIs keep the same as the original repo. Instead of the following changes:

```python
# Old
occupancy_grid = OccupancyGrid(256, radius, [0, 0, 0])
# New
occupancy_grid = OccupancyGrid(256, radius, torch.tensor([0, 0, 0], dtype=torch.float, device='cuda'))

# Old
bounding_sphere = Sphere(radius, [0, 0, 0])
# New
bounding_sphere = Sphere(radius, torch.tensor([0, 0, 0], dtype=torch.float, device='cuda'))
```

### Additional features

Some additional functionalities are extended compared to the original repo, including:

- `VolumeRendering.integrate_with_weights` now supports not only 3-channel RGB, but also 1-channel depth and 16, 32, 48, 64-channel feature vectors
- New function: `OccupancyGrid::update_with_sdf_positions`, which supports directly updating the SDF value in the occupancy according to the specified position

## Acknowledgements

Most implementations are derived from the original repo. Thanks for Radu Alexandru's great work.