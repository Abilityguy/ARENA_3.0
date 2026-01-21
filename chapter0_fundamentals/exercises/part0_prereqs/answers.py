# %%

import math
import os
import sys
from pathlib import Path

import einops
import numpy as np
import torch as t
from torch import Tensor

# Make sure exercises are in the path
chapter = "chapter0_fundamentals"
section = "part0_prereqs"
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

import part0_prereqs.tests as tests
from part0_prereqs.utils import display_array_as_img, display_soln_array_as_img

MAIN = __name__ == "__main__"
# %%
arr = np.load(section_dir / "numbers.npy")
# %%
print(arr[0].shape)
display_array_as_img(arr[0])  # plotting the first image in the batch
# %%
arr_stacked = einops.rearrange(arr, "b c h w -> c h (b w)")
print(arr_stacked.shape)
display_array_as_img(arr_stacked)  # plotting all images, stacked in a row
# %%
# (1) Column-stacking
# Your code here - define arr1
arr1 = einops.rearrange(arr, "b c h w -> c (b h) w")

display_array_as_img(arr1)
# %%
# (2) Column-stacking and copying
# Your code here - define arr2
arr2 = einops.repeat(arr[0], "c h w -> c (2 h) w")
display_array_as_img(arr2)
# %%
# (3) Row-stacking and double-copying
# Your code here - define arr3
arr3 = einops.repeat(arr[0: 2], "b c h w -> c (b h) (2 w)")
display_array_as_img(arr3)
# %%
# (4) Stretching
# Your code here - define arr4
arr4 = einops.repeat(arr[0], "c h w -> c (h 2) w")
display_array_as_img(arr4)
# %%
# (5) Split channels
# Your code here - define arr5
arr5 = einops.rearrange(arr[0], "c h w -> h (c w)")
display_array_as_img(arr5)
# %%
# (6) Stack into rows & cols
# Your code here - define arr6
arr6 = einops.rearrange(arr, "(b1 b2) c h w -> c (b1 h) (b2 w)", b1=2, b2=3)
display_array_as_img(arr6)
# %%
# (7) Transpose
# Your code here - define arr7
arr7 = einops.rearrange(arr[1], "c h w -> c w h")
display_array_as_img(arr7)
# %%
# (8) Shrinking
# Your code here - define arr8
arr8 = einops.reduce(arr, "(b1 b2) c (h h2) (w w2) -> c (b1 h) (b2 w)", "max", h2=2, w2=2, b1=2)
display_array_as_img(arr8)
# %%
