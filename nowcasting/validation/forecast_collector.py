import matplotlib.pyplot as plt
from matplotlib import pyplot, colors
import numpy as np
import pandas as pd
import torch
import zarr
from omegaconf import OmegaConf
from pathlib import Path
import json
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from tqdm import tqdm
import warnings

@dataclass
class SampleMetadata:
    """Metadata for a saved sample"""
    sample_id: int
    intensity_score: float
    batch_idx: int
    original_batch_position: int
    timestamp: Optional[str] = None

import json
import numpy as np
import torch
import zarr
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

class ForecastCollector:
    """
    Collects forecast samples during validation and saves high-intensity cases
    for later visualization. Optimized to write all samples at once during finalize.
    """
    
    def __init__(
        self,
        save_dir: Union[str, Path],
        model_name: str,
        data_info: Dict,
        intensity_threshold: float = 0.8,
        max_samples: int = 100,
        save_ensemble: bool = False,
        eval_subdir: str = "evaluations",
        enforce_registry_match: bool = True,
    ):
        """
        Initialize the forecast collector.
        
        Args:
            save_dir: Directory to save results
            model_name: Name identifier for this model
            data_info: Channel information with mean/std statistics
            intensity_threshold: Minimum intensity to save sample (quantile)
            max_samples: Maximum number of samples to save
            save_ensemble: If True, save full ensemble; if False, save ensemble mean
            enforce_registry_match: If True, only save samples whose indices
                                    match those in the global registry.
        """
        self.save_dir = Path(save_dir)
        self.model_name = model_name
        self.data_info = data_info
        self.intensity_threshold = intensity_threshold
        self.max_samples = max_samples
        self.save_ensemble = save_ensemble
        self.eval_subdir = eval_subdir
        self.enforce_registry_match = enforce_registry_match # STORE THE NEW PARAMETER
        
        # Create directories
        self.model_dir = self.save_dir / self.eval_subdir / model_name
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory storage for samples
        self.samples = {
            'targets': [],
            'predictions': [],
            'intensities': [],
            'indices': [],
        }
        self.batch_counter = 0
        
        # For consistency across models
        self.global_sample_registry = self.save_dir / self.eval_subdir / "sample_registry.json"
        self.selected_samples = self._load_or_create_sample_registry()

        # If enforcing registry match, convert the list of indices to a set for faster lookup
        if self.enforce_registry_match and self.selected_samples["selected_samples"]:
            self.registry_indices_set = set(tuple(i) for i in self.selected_samples["selected_samples"])
        else:
            self.registry_indices_set = None

    def _load_or_create_sample_registry(self) -> Dict:
        """Load existing sample registry or create a new one."""
        if self.global_sample_registry.exists():
            with open(self.global_sample_registry, 'r') as f:
                try:
                    registry = json.load(f)
                    # Check for "selected_samples" and "metadata" keys
                    if "selected_samples" in registry and "metadata" in registry:
                        return registry
                    else:
                        print(f"Warning: Existing registry file {self.global_sample_registry} has invalid format. Creating new.")
                except json.JSONDecodeError:
                    print(f"Warning: Existing registry file {self.global_sample_registry} is corrupted. Creating new.")
        return {"selected_samples": [], "metadata": {}}
    
    def _save_sample_registry(self):
        """Save the sample registry to disk."""
        with open(self.global_sample_registry, 'w') as f:
            json.dump(self.selected_samples, f, indent=2)

    def _to_numpy(self, array):
        """Convert torch.Tensor or np.ndarray to np.ndarray (on CPU)."""
        if isinstance(array, torch.Tensor):
            return array.detach().cpu().numpy()
        elif isinstance(array, np.ndarray):
            return array
        else:
            raise TypeError(f"Unsupported type {type(array)}")
    
    def _calculate_intensity(self, y_target):
        """
        Calculate intensity score for each sample in the batch.
        Works for torch.Tensor or np.ndarray.
        """
        if isinstance(y_target, torch.Tensor):
            # [B, ...] -> flatten across all but batch
            return y_target.flatten(start_dim=1).mean(dim=1)
        elif isinstance(y_target, np.ndarray):
            return y_target.reshape(y_target.shape[0], -1).mean(axis=1)
        else:
            raise TypeError(f"Unsupported type {type(y_target)}")
    
    def _should_save_sample(self, sample_id: str, intensity_score: float, sample_index: Optional[Tuple[int, int, int]] = None) -> bool:
        """Determine if a sample should be saved"""
        if self.enforce_registry_match:
            # Check if index is in the registry set
            if sample_index is not None and self.registry_indices_set is not None:
                return sample_index in self.registry_indices_set
            # If no registry is found, enforce_registry_match is meaningless, fall back to intensity
            return intensity_score >= self.intensity_threshold
        else:
            # Use registry if available (original logic)
            if "selected_samples" in self.selected_samples and self.selected_samples["selected_samples"]:
                return sample_id in self.selected_samples["selected_samples"]
            
            # Otherwise use intensity threshold
            return intensity_score >= self.intensity_threshold
    
    def process_batch(
        self,
        y_target: torch.Tensor,
        y_pred: torch.Tensor,
        sample_idx: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    ):
        """
        Process a batch of predictions, selecting high-intensity samples.
        
        Args:
            y_target: Target tensor [B, C, T, H, W]
            y_pred: Prediction tensor [B, C, T, H, W] or [B, C, T, H, W, E]
            sample_idx: Tuple of (t_idx, h_idx, w_idx), each shape [B]
        """
        # Skip only if max_samples is set and already full
        if self.max_samples is not None and len(self.samples['targets']) >= self.max_samples:
            return
            
        batch_size = y_target.shape[0]

        intensities = self._calculate_intensity(y_target)

        # Handle ensemble dimension
        if not self.save_ensemble and (
            (isinstance(y_pred, torch.Tensor) and y_pred.dim() == 6) or 
            (isinstance(y_pred, np.ndarray) and y_pred.ndim == 6)
        ):
            if isinstance(y_pred, torch.Tensor):
                y_pred = y_pred.mean(dim=-1)
            else:
                y_pred = y_pred.mean(axis=-1)
        
        # Move to CPU/numpy
        y_target = self._to_numpy(y_target)
        y_pred = self._to_numpy(y_pred)
        intensities = self._to_numpy(intensities)
        
        # If sample indices are provided, also move to numpy
        sample_idx_np = None
        if sample_idx is not None:
            try:
                sample_idx_np = tuple(self._to_numpy(idx) for idx in sample_idx)
            except Exception:
                sample_idx_np = None

        # Loop over samples
        for i in range(batch_size):
            if self.max_samples is not None and len(self.samples['targets']) >= self.max_samples:
                break
            
            # Get the index of the current sample
            current_idx = None
            if sample_idx_np is not None and len(sample_idx_np) >= 3:
                try:
                    current_idx = (
                        int(sample_idx_np[0][i]),
                        int(sample_idx_np[1][i]),
                        int(sample_idx_np[2][i]),
                    )
                except (TypeError, ValueError, IndexError):
                    current_idx = None
            if current_idx is None:
                # Stable synthetic index for dataloaders that do not emit sample_idx.
                current_idx = (int(self.batch_counter), int(i), -1)
            
            # The original logic for `sample_id` is based on batch counter; this is for internal tracking.
            # The registry-based check will use the `current_idx`.
            sample_id = f"{self.batch_counter:06d}_{i:03d}"
            intensity_score = float(intensities[i])
            
            # Use the new check, passing the actual sample index
            if self._should_save_sample(sample_id, intensity_score, sample_index=current_idx):
                self.samples['targets'].append(y_target[i])
                self.samples['predictions'].append(y_pred[i])
                self.samples['intensities'].append(intensity_score)
                self.samples['indices'].append(current_idx) # Append the actual index
        
        self.batch_counter += 1

    def finalize(self):
        """Write all collected samples to disk at once."""
        if not self.samples['targets']:
            print("No samples collected.")
            return

        # NEW LOGIC: If a registry doesn't exist, create it from the first model's indices
        if not self.global_sample_registry.exists() or not self.selected_samples["selected_samples"]:
            self.selected_samples["selected_samples"] = [list(idx) for idx in self.samples['indices'] if idx is not None]
            self.selected_samples["metadata"]["model"] = self.model_name
            self._save_sample_registry()
            print(f"Created global sample registry with {len(self.samples['indices'])} indices.")
        
        zarr_path = self.model_dir / f"{self.model_name}.zarr"
        store = zarr.open_group(str(zarr_path), mode='w')
        n_samples = len(self.samples['targets'])
        
        target_shape = self.samples['targets'][0].shape
        prediction_shape = self.samples['predictions'][0].shape
        
        store.create_array('targets', shape=(n_samples, *target_shape), dtype=np.float16)
        store.create_array('predictions', shape=(n_samples, *prediction_shape), dtype=np.float16)
        store.create_array('intensities', shape=(n_samples,), dtype=np.float16)
        store.create_array('indices', shape=(n_samples, 3), dtype=np.int32)
        
        for i in range(n_samples):
            store['targets'][i] = np.asarray(self.samples['targets'][i], dtype=np.float16)
            store['predictions'][i] = np.asarray(self.samples['predictions'][i], dtype=np.float16)
            store['intensities'][i] = np.asarray(self.samples['intensities'][i], dtype=np.float16)
            store['indices'][i] = np.asarray(self.samples['indices'][i], dtype=np.int32)
        
        metadata_path = self.model_dir / "metadata.json"
        metadata_dict = {
            "model_name": self.model_name,
            "total_samples": n_samples,
            "data_info": self.data_info,
            "save_ensemble": self.save_ensemble,
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata_dict, f, indent=2)
        
        print(f"Saved {n_samples} samples for model {self.model_name}")

def flatten_omegaconf_list(l):
    """Recursively flatten a ListConfig or nested lists into a flat Python list."""
    flat_list = []
    for item in l:
        # OmegaConf Lists behave like sequences, so check recursively
        if OmegaConf.is_list(item) or isinstance(item, list):
            flat_list.extend(flatten_omegaconf_list(item))
        else:
            flat_list.append(item)
    return flat_list


def extract_data_info_from_config(config, use_input_vars=True):
    """
    Extract data_info from OmegaConf config structure.
    
    Args:
        config: OmegaConf config object with dataloader section
        use_input_vars: If True, use in_vars; if False, use out_vars
    
    Returns:
        dict: data_info dictionary with channel statistics
    """
    dataloader_config = config.dataloader
    var_list = dataloader_config.var_info.in_vars if use_input_vars else dataloader_config.var_info.out_vars
    transforms = dataloader_config.var_info.transforms

    # Flatten the var_list to handle nested ListConfigs
    flat_var_list = flatten_omegaconf_list(var_list)

    data_info = {}

    for var_name in flat_var_list:
        # Check if this variable has normalization transforms
        if var_name in transforms and 'normalize' in transforms[var_name]:
            norm_stats = transforms[var_name]['normalize']
            data_info[var_name] = {
                'mean': norm_stats['mean'],
                'std': norm_stats['std']
            }
        else:
            # Check for group-level transforms (e.g., 'radar')
            var_group = var_name.split('/')[0]
            if var_group in transforms:
                group_transforms = transforms[var_group]
                if 'default_rainrate' in group_transforms:
                    # For radar data
                    norm_stats = group_transforms['default_rainrate']
                    data_info[var_name] = {
                        'mean': norm_stats['mean'],
                        'std': norm_stats['std']
                    }
                elif 'normalize' in group_transforms:
                    # For other group-level normalizations
                    norm_stats = group_transforms['normalize']
                    data_info[var_name] = {
                        'mean': norm_stats['mean'],
                        'std': norm_stats['std']
                    }
                else:
                    print(f"Warning: No normalization stats found for {var_name}")
                    data_info[var_name] = {'mean': 0.0, 'std': 1.0}
            else:
                print(f"Warning: No normalization stats found for {var_name}")
                data_info[var_name] = {'mean': 0.0, 'std': 1.0}

    return data_info

