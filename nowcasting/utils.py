# ==============================================================================
# Original Code Source:
# This code is a slightly modified version from the repository accompanying
# the paper "Improving precipitation nowcasting for high-intensity events using deep 
# generative models with balanced loss and temperature data: a case study in the Netherlands" by
# Charlotte Van Nooten and colleagues.
# Repository: https://github.com/charlottecvn/precipitationnowcasting-generativemodels-highevents
# ==============================================================================

import numpy as np
import torch

def r_to_dbz(r):
    """Convert mm/h to dBZ in place, skipping zeros."""
    if isinstance(r, np.ndarray):
        mask = r != 0  # Mask to exclude zero elements
        r[mask] = 10 * np.log10(200*r[mask]**(8/5)+1) 
    elif isinstance(r, torch.Tensor):
        mask = r != 0
        result = 10 * torch.log10(200*r[mask]**(8/5)+1) 
        r[mask] = result.to(r.dtype)
    else:
        raise TypeError('Array must be a numpy array or torch Tensor.')

def dbz_to_r(dbz):
    """Convert dBZ to mm/h in place, skipping zeros."""
    if isinstance(dbz, np.ndarray):
        mask = dbz != 0  # Mask to exclude zero elements
        dbz[mask] = ((10**(dbz[mask]/10)-1)/200)**(5/8)
    elif isinstance(dbz, torch.Tensor):
        mask = dbz != 0
        result = ((10**(dbz[mask]/10)-1)/200)**(5/8)
        dbz[mask] = result.to(dbz.dtype)
    else:
        raise TypeError('Array must be a numpy array or torch Tensor.')

def clipping(x, min_val=0, max_val=100):
    # Clip depending on array type
    if isinstance(x, np.ndarray):
        np.clip(x, min_val, max_val, out=x)
    elif isinstance(x, torch.Tensor):
        x.clamp_(min_val, max_val)
    else:
        raise TypeError('Input must be a numpy array or torch Tensor.')

def data_prep(x, norm_method='minmax', convert_to_dbz=False, undo=False):
    """Perform in-place data preprocessing with optimized speed, skipping zeros."""
    assert norm_method in ('minmax', 'minmax_tanh')

    MIN, MAX = 0, 100
    if convert_to_dbz:
        MAX = 55

    if not undo:
        if convert_to_dbz:
            r_to_dbz(x)  # Convert in place, skipping zeros

        clipping(x, min_val=MIN, max_val=MAX)

        if norm_method == 'minmax_tanh':
            x -= (MIN + MAX / 2)
            x /= (MAX / 2 - MIN)
        else:
            x -= MIN
            x /= (MAX - MIN)
    else:
        clipping(x, min_val=0, max_val=1)

        if norm_method == 'minmax_tanh':
            x *= (MAX / 2 - MIN)
            x += (MIN + MAX / 2)
        else:
            x *= (MAX - MIN)
            x += MIN

        if convert_to_dbz:
            dbz_to_r(x)  # Modify x in place, skipping zeros