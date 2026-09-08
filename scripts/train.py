import importlib
import numpy as np
import os
import re
import shutil
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

from codecarbon import EmissionsTracker

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.utilities.rank_zero import rank_zero_only

import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy("file_system")

from fire import Fire
from omegaconf import OmegaConf

def _to_dotlist_value(value: Any) -> str:
    """
    Convert a Python value from Fire kwargs to an OmegaConf dotlist literal.
    """
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        lower = value.lower()
        if lower == "null":
            return "null"
        if lower == "true":
            return "true"
        if lower == "false":
            return "false"
        # Quote strings so OmegaConf parses them as string scalars.
        return f'"{value}"'
    return f'"{str(value)}"'

@rank_zero_only
def copy_config_to_log_dir(config_path: Optional[str], trainer: Trainer):
    """Copy the run YAML into the TensorBoard log directory (rank 0 only)."""
    if config_path and trainer.logger:
        tb_dir = trainer.logger.log_dir
        os.makedirs(tb_dir, exist_ok=True)
        shutil.copy(
            config_path,
            os.path.join(tb_dir, os.path.basename(config_path))
        )
        print(f"Copied config {config_path} → {tb_dir}")

def get_latest_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    """Return the most recently written last*.ckpt in checkpoint_dir.

    Picks by mtime, not version suffix (resume can leave stale last-vN.ckpt).
    """
    checkpoint_dir = Path(checkpoint_dir)
    pattern = re.compile(r"last(?:-v(\d+))?\.ckpt$")

    latest_file = None
    latest_mtime = -1.0

    for file in checkpoint_dir.glob("last*.ckpt"):
        if not pattern.fullmatch(file.name):
            continue
        mtime = file.stat().st_mtime
        if mtime > latest_mtime:
            latest_mtime = mtime
            latest_file = file

    return latest_file

def load_version(version_path: str) -> Tuple[OmegaConf, TensorBoardLogger, Optional[Path]]:
    """Reload config, TensorBoard logger, and last checkpoint from a Lightning version dir."""
    version_path = Path(version_path)

    # Extract components from the version_path
    # Example: /logs/experiment_name/version_0
    version = int(version_path.name.replace("version_", ""))
    experiment_name = version_path.parent.name
    logger_dir = version_path.parent.parent

    # Reconstruct the TensorBoardLogger
    logger = TensorBoardLogger(logger_dir, name=experiment_name, version=version)

    # Find the latest checkpoint within this version's checkpoint directory
    checkpoint_dir = os.path.join(logger.log_dir, "checkpoints")
    resume_checkpoint = get_latest_checkpoint(checkpoint_dir)

    # Try to find the original config file used for this version.
    config_files = list(Path(logger.log_dir).glob("*.yaml"))
    config_file = next((f for f in config_files if f.name != "hparams.yaml" and f.stat().st_size > 0), None)

    print(f"Loaded config from: {config_file}")
    if config_file is not None:
        config = OmegaConf.load(config_file)
    else:
        # Fallback if no config file is found (though it should ideally exist)
        print("Warning: No config file found in the version directory. Using empty config.")
        config = OmegaConf.create()

    return config, logger, resume_checkpoint

def load_module(path: str):
    """Import a module, or an object from a dotted path such as `pkg.mod.Class`."""
    try:
        # First, try to import it as a full module
        return importlib.import_module(path)
    except ModuleNotFoundError as e:
        # If it fails, split into module + object
        if "." not in path:
            raise ImportError(f"Invalid import path: '{path}' is not a module or object path.")

        module_path, object_name = path.rsplit(".", 1)
        try:
            module = importlib.import_module(module_path)
            return getattr(module, object_name)
        except (ModuleNotFoundError, AttributeError) as e2:
            raise ImportError(
                f"Could not import '{object_name}' from module '{module_path}': {e2}"
            ) from e2

def setup_data(config: OmegaConf):
    """Build the Lightning DataModule from `config.module_path` and remaining kwargs."""
    print(f"Setting up data module from: {config.module_path}")
    data_module = load_module(config.pop('module_path'))
    data_module = data_module(**config)
    data_module.setup()
    print("Data module setup complete.")
    return data_module

def setup_model(config: OmegaConf):
    """
    Initializes a PyTorch Lightning model.

    If a checkpoint path is provided, only the model weights are loaded
    (without restoring trainer/optimizer state).
    Optionally compiles the model if 'compile' is set in the config.
    """
    print(f"Setting up model from: {config.model.module_path}")
    Model = load_module(config.model.pop("module_path"))

    # Pop flags from config
    compile_model = config.model.pop("compile", False)
    checkpoint_path = config.model.pop("checkpoint_path", None)
    initial_weights = config.model.pop("initial_weights", None)

    if checkpoint_path:
        print(f"Loading model weights from checkpoint: {checkpoint_path}")
        model = Model.load_from_checkpoint(checkpoint_path, config=config)
    else:
        model = Model(config)

    if initial_weights is not None:
            print(f"Loading weights from {initial_weights}...")

            weights = torch.load(initial_weights, map_location=model.device)

            # Handle Lightning checkpoint format (state_dict nested under key)
            if "state_dict" in weights:
                weights = weights["state_dict"]

            filtered_weights = {
                k: v for k, v in weights.items()
                if not (k.startswith("autoencoder.") or k.startswith("context_encoder.autoencoder."))
            }

            model.load_state_dict(
                filtered_weights,
                strict=False
            )

    if compile_model:
        print("Compiling model with torch.compile...")
        model = torch.compile(model, mode="default")

    print("Model setup complete.")
    return model

def setup_trainer(
    config_trainer: Optional[OmegaConf] = None, 
    config_callbacks: Optional[OmegaConf] = None,
    config_logger: Optional[OmegaConf] = None,
    logger: Optional[TensorBoardLogger] = None
) -> Trainer:
    """Build a Lightning Trainer from YAML blocks for trainer, callbacks, and logger."""
    print("Setting up PyTorch Lightning Trainer...")
    if logger is None:
        logger = TensorBoardLogger(**config_logger) if config_logger else None
    else:
        print(f"TensorBoard Logger Version: {logger.version}")
    
    callbacks = []
    if config_callbacks is not None:
        # Add additional callbacks here
        if 'checkpoint' in config_callbacks:
            checkpoint_callback = ModelCheckpoint(**config_callbacks['checkpoint'])
            callbacks.append(checkpoint_callback)

        if 'early_stopping' in config_callbacks:
            early_stopping_callback = EarlyStopping(**config_callbacks['early_stopping'])
            callbacks.append(early_stopping_callback)

    trainer = Trainer(
        **config_trainer,
        logger=logger,
        callbacks=callbacks,
    )

    return trainer

def main(config: Optional[str] = None, version_path: Optional[str] = None, seed: Optional[int] = None, name_suffix: Optional[str] = None, **kwargs):
    """Train from a YAML config, or resume from a Lightning `version_X` directory.

    Extra Fire kwargs override config keys, e.g. `trainer.max_epochs=10`.
    """
    print("Starting main training process.")

    resume_checkpoint = None
    logger = None
    if version_path:
        # True resume
        config, logger, resume_checkpoint = load_version(version_path)
        config.model.pop("checkpoint_path", None)
        model = setup_model(config)  # fresh init, trainer restores
    else:
        config_path = config
        config = OmegaConf.load(config) if (config is not None) else OmegaConf.create()
        if kwargs:
            # merge CLI overrides (trainer.max_epochs=10, etc.)
            dotlist = [f"{k}={_to_dotlist_value(v)}" for k, v in kwargs.items()]
            overrides = OmegaConf.from_dotlist(dotlist)
            config = OmegaConf.merge(config, overrides)

        if name_suffix:
            config.logger.name = f"{config.logger.name}{name_suffix}"

        if seed is not None:
            seed_everything(int(seed), workers=True)
            print(f"[seed_replication] seed_everything({seed}, workers=True) -> logger.name={config.logger.name}", flush=True)

        model = setup_model(config)

    data_module = setup_data(config.dataloader)

    trainer = setup_trainer(
                            config.trainer, 
                            config.callbacks, 
                            config.logger,
                            logger=logger
                            )

    if not version_path:
        copy_config_to_log_dir(config_path, trainer)

    os.makedirs("results/environment_metrics", exist_ok=True)
    tracker = EmissionsTracker(
            project_name='Earthformer',
            output_dir="results/environment_metrics",
            log_level="warning",
            allow_multiple_runs=True,
        )
    tracker.start()

    print("Initiating model training...")
    trainer.fit(model, data_module, ckpt_path=resume_checkpoint)
    print("Model training finished.")

    emissions = tracker.stop()
    print(f"Emissions for this run: {emissions} kg CO2")
    
if __name__ == "__main__":
    Fire(main)