import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib import colors as mcolors
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# Given Predefined prime numbers for hashing
PRIMES = [1, 2654435761, 805459861, 492876847, 3346619431]

def num_params(model):
    return sum([np.prod(p.size()) for p in model.parameters()])
  
# helper functions to add noise and normalize the image
def add_noise(img, sigma):
    # add noise to the image
    noise = torch.randn_like(img) * sigma
    return img + noise

def normalize(img, min=None, max=None):
    if min is not None and max is not None:
        return (img - min) / (max - min)
    return (img - img.min()) / (img.max() - img.min())

def unnormalize(img, min, max):
    return img * (max - min) + min 


# function to visualise a deformation field
def showFlow(def_x):
    x = def_x.squeeze().numpy()[0, :, :]
    y = def_x.squeeze().numpy()[1, :, :]
    #show flow map for numpy
    H, W = x.shape
    rho = np.sqrt(x * x + y * y)
    theta = np.arctan2(x, -y)
    theta2 = (-theta + np.pi) / (2.0 * np.pi)
    rho = np.clip(rho / np.percentile(rho, 99), 0, 1)
    hsv = np.stack((theta2, rho, np.ones((H, W))), axis=2)
    rgb = mcolors.hsv_to_rgb(hsv)
    return rgb

def plot_comparison(gt_img, images: list, names: list):
    fig, ax = plt.subplots(1, len(images), figsize=(20, 10))
    for i in range(len(images)):
      # calculate the ssim and psnr values
      ssim_val = ssim(normalize(gt_img).numpy(), normalize(images[i]).numpy(), data_range=1)
      psnr_val = psnr(normalize(gt_img).numpy(), normalize(images[i]).numpy())
      ax[i].imshow(torch.clamp(images[i], -500, 500), cmap='gray')
      ax[i].set_title(f'{names[i]} - SSIM: {ssim_val:.4f}, PSNR: {psnr_val:.4f}')
      ax[i].axis('off')
    plt.show()


# Given: Fast hash function
@torch.no_grad()
def fast_hash(indices: torch.Tensor, primes: torch.Tensor, hashmap_size: int) -> torch.Tensor:
    """
    Fast hash function based on prime multiplication and XOR folding.
    """
    d = indices.shape[-1]  # Dimensionality
    indices = (indices * primes[:d]) & 0xffffffff  # Convert to uint32
    for i in range(1, d):
        indices[..., 0] ^= indices[..., i]  # XOR folding
    return indices[..., 0] % hashmap_size  # Modulo hashmap size


class MultiResHashGrid(nn.Module):
  def __init__(
    self,
    dim: int,
    n_levels: int = 16,
    n_features_per_level: int = 2,
    log2_hashmap_size: int = 15,
    base_resolution: int = 16,
    finest_resolution: int = 512,
    _HashGrid: torch.nn.Module = None):
    """NVidia's hash grid encoding
    https://nvlabs.github.io/instant-ngp/

    The output dimensions is `n_levels` * `n_features_per_level`,
    or your can simply access `model.output_dim` to get the output dimensions

    Args:
      dim (int): input dimensions, supports at most 7D data.
      n_levels (int, optional): number of grid levels. Defaults to 16.
      n_features_per_level (int, optional): number of features per grid level.
        Defaults to 2.
      log2_hashmap_size (int, optional): maximum size of the hashmap of each
        level in log2 scale. According to the paper, this value can be set to
        14 ~ 24 depending on your problem size. Defaults to 15.
      base_resolution (int, optional): coarsest grid resolution. Defaults to 16.
      finest_resolution (int, optional): finest grid resolution. According to
        the paper, this value can be set to 512 ~ 524288. Defaults to 512.
    """
    super().__init__()
    self.dim = dim
    self.n_levels = n_levels
    self.n_features_per_level = n_features_per_level
    self.log2_hashmap_size = log2_hashmap_size
    self.base_resolution = base_resolution
    self.finest_resolution = finest_resolution

    # from paper eq (3)
    b = math.exp((math.log(finest_resolution) - math.log(base_resolution))/(n_levels-1))

    levels = []
    for level_idx in range(n_levels):
      # The resolution of the grid is calculated given the base resolution and the level index
      resolution = math.floor(base_resolution * (b ** level_idx))
      hashmap_size = min(resolution ** dim, 2 ** log2_hashmap_size)
      levels.append(_HashGrid(
        dim = dim,
        n_features = n_features_per_level,
        hashmap_size = hashmap_size,
        resolution = resolution
      ))
    self.levels = nn.ModuleList(levels)

    self.input_dim = dim
    self.output_dim = n_levels * n_features_per_level

  def forward(self, x: torch.Tensor):
    return torch.cat([level(x) for level in self.levels], dim=-1)

def interpolate_weights(coords: torch.Tensor, binary_mask: torch.Tensor):
  coords_shape = coords.shape[:-1]
  coords_integer = coords.to(torch.int32)
  coords_fractional = coords - coords_integer.float().detach()
  coords_integer = coords_integer.unsqueeze(dim=-2) # shape: (H*W, 1, dim)
  coords_fractional = coords_fractional.unsqueeze(dim=-2) # shape: (H*W, 1, dim)
  # to match the input batch shape
  binary_mask = binary_mask.reshape((1,)*coords_shape + binary_mask.shape) # shape: (1, num_neighbors, dim)
  # get neighbors' indices and weights on each dim
  indices = torch.where(binary_mask, coords_integer, coords_integer+1) # shape: (H*W, num_neighbors, dim)
  weights = torch.where(binary_mask, 1-coords_fractional, coords_fractional) # shape: (H*W, num_neighbors, dim)
  # aggregate nehgibors' interp weights
  weights = weights.prod(dim=-1, keepdim=True) # shape: (H*W, num_neighbors, 1)
  return indices, weights

class HashGridEncoding(nn.Module):
  def __init__(
    self,
    dim: int,
    n_levels: int = 16,
    n_features_per_level: int = 2,
    log2_hashmap_size: int = 15,
    base_resolution: int = 16,
    finest_resolution: int = 512,
    ):
    """NVidia's hash grid encoding
    https://nvlabs.github.io/instant-ngp/

    The output dimensions is `n_levels` * `n_features_per_level`,
    or your can simply access `model.output_dim` to get the output dimensions

    Args:
      dim (int): input dimensions, supports at most 7D data.
      n_levels (int, optional): number of grid levels. Defaults to 16.
      n_features_per_level (int, optional): number of features per grid level.
        Defaults to 2.
      log2_hashmap_size (int, optional): maximum size of the hashmap of each
        level in log2 scale. According to the paper, this value can be set to
        14 ~ 24 depending on your problem size. Defaults to 15.
      base_resolution (int, optional): coarsest grid resolution. Defaults to 16.
      finest_resolution (int, optional): finest grid resolution. According to
        the paper, this value can be set to 512 ~ 524288. Defaults to 512.
    """
    super().__init__()
    self.dim = dim
    self.n_levels = n_levels
    self.n_features_per_level = n_features_per_level
    self.log2_hashmap_size = log2_hashmap_size
    self.base_resolution = base_resolution
    self.finest_resolution = finest_resolution

    # from paper eq (3)
    self.b = math.exp((math.log(finest_resolution) - math.log(base_resolution))/(n_levels-1))

    levels = []
    self.resolutions = []
    for level_idx in range(n_levels):
      # The resolution of the grid is calculated given the base resolution and the level index
      self.resolutions.append(math.floor(self.base_resolution * (self.b ** level_idx)))
      levels.append(nn.Embedding(num_embeddings=self.log2_hashmap_size, embedding_dim=self.n_features_per_level
      ))
      
    self.levels = nn.ModuleList(levels)
    self.initialize_weights()
  
    self.input_dim = dim
    self.output_dim = n_levels * n_features_per_level
  
    # Given: Predefine primes for hashing and register as buffer
    self.primes = torch.tensor(PRIMES, dtype=torch.int64)

    # Given: Precompute binary masks for neighbors
    num_neighbors = 1 << self.dim
    neighbors = np.arange(num_neighbors, dtype=np.int64).reshape((-1, 1))
    dimensions = np.arange(self.dim, dtype=np.int64).reshape((1, -1))
    # Obtain the mask for each dimension by checking the bit at the position of the dimension in the neighbors index 
    self.binary_mask = torch.tensor(neighbors & (1 << dimensions) == 0, dtype=bool) # shape: (num_neighbors, dim)
  
  def initialize_weights(self):
    # initialize the weights of the embedding layers
    for level in self.levels:
      nn.init.uniform_(level.weight, a=-0.0001, b=0.0001)
      
  def forward(self, x: torch.Tensor):
      output = []
      for level_idx, level in enumerate(self.levels):
        resolution = self.resolutions[level_idx]
        
        x_rescaled = x * resolution
        hashmap_size = min(resolution ** self.dim, 2 ** self.log2_hashmap_size)
        weights, indices = interpolate_weights(x_rescaled, self.binary_mask)
        # hash the indices
        indices = fast_hash(indices, self.primes, hashmap_size)
        # get the features from the embedding layer
        features = level(indices)
        # aggregate the features
        output.append((features * weights).sum(dim=1))
      return torch.cat(output, dim=-1)
  