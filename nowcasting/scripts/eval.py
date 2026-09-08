import importlib
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import List, Dict, Any, Optional
from pathlib import Path

from fire import Fire
from omegaconf import OmegaConf

from nowcasting.validation.validation import PrecipitationEvaluator
from nowcasting.validation.forecast_collector import ForecastCollector, extract_data_info_from_config

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

def evaluation(
    model,
    dataloader: torch.utils.data.DataLoader,
    config,
    model_name: str = "pysteps",
    save_dir: str = "./results",
    eval_subdir: str = "evaluations",
    num_samples: Optional[int] = None,
    leadtimes: List[int] = [5, 10, 15, 30, 60, 90],
    thresholds: List[float] = [0, 0.5, 1, 2, 5, 10, 20, 30],
    crop_size: Optional[List[int]] = None,
    compute_dry_to_wet: bool = False,
    dry_threshold: float = 0.5,
):
    """Run verification metrics on `dataloader` and write scores under `save_dir`."""
    if num_samples is None:
        num_batches = float('inf')
    else:
        num_batches = num_samples // dataloader.batch_size

    evaluator = PrecipitationEvaluator(
        nowcast_method=model_name,
        # metrics_to_compute=['pod', 'far', 'csi', 'bias','mse', 'mae', 'rmse', 'bias_score', 'correlation','fss'],
        save_dir=save_dir,
        eval_subdir=eval_subdir,
        thresholds=thresholds,
        leadtimes=leadtimes,
        crop_size=tuple(crop_size) if crop_size is not None else None,
        save_frequency=100,
        compute_crps=False,
        compute_dry_to_wet=compute_dry_to_wet,
        dry_threshold=dry_threshold,
        # device=getattr(model, 'device', 'cpu')
    )

    data_info = extract_data_info_from_config(config)
    collector = ForecastCollector(
        save_dir=save_dir,
        model_name=model_name,
        intensity_threshold=0.8,
        data_info=data_info,
        max_samples=100,
        save_ensemble=False,  # Save ensemble mean
        eval_subdir=eval_subdir
    )

    for idx, batch in tqdm(enumerate(dataloader), desc="Verifying batches"):
        if len(batch) == 3:
            x, y, sample_idx = batch
        elif len(batch) == 2:
            x, y = batch
            sample_idx = None
        else:
            raise ValueError(
                f"Unexpected batch structure with {len(batch)} items; expected 2 or 3."
            )
        # model(x, y) -> (y, y_hat) with shared post-processing
        input_radar_mmh = None
        if compute_dry_to_wet and hasattr(model, "to_mmh"):
            last_radar = x[:, 0, -1, ...]
            if hasattr(model, "device"):
                last_radar = last_radar.to(model.device)
            input_radar_mmh = model.to_mmh(last_radar)

        y, y_hat = model(x, y)

        evaluator.process_batch(y, y_hat, input_radar_mmh=input_radar_mmh)
        collector.process_batch(y, y_hat, sample_idx=sample_idx)

        if idx >= num_batches:
            break

    evaluator.save_results("final_results")
    try:
        collector.finalize()
    except Exception as exc:
        print(f"Warning: ForecastCollector finalize failed: {exc}")

    scores = evaluator.get_final_scores()
    print(f"\nGenerated {len(scores)} metric categories")
    for metric_type, data in scores.items():
        print(f"  {metric_type}: {len(data) if data else 0} entries")
    _save_eval_plots(scores=scores, save_dir=save_dir, eval_subdir=eval_subdir, model_name=model_name)

def _save_eval_plots(scores: Dict[str, Any], save_dir: str, eval_subdir: str, model_name: str) -> None:
    """Create compact summary plots from evaluation results."""
    out_dir = Path(save_dir) / eval_subdir / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    cont = scores.get("continuous", [])
    cat = scores.get("categorical", [])
    if not cont:
        print("Warning: no continuous scores available for plotting.")
        return

    cont_sorted = sorted(cont, key=lambda x: int(x["leadtime"]))
    lead = [int(x["leadtime"]) for x in cont_sorted]
    mse = [float(x["MSE"]) for x in cont_sorted]
    mae = [float(x["MAE"]) for x in cont_sorted]
    rmse = [float(x["RMSE"]) for x in cont_sorted]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(lead, mse, marker="o", label="MSE")
    ax.plot(lead, mae, marker="o", label="MAE")
    ax.plot(lead, rmse, marker="o", label="RMSE")
    ax.set_xlabel("Lead time (min)")
    ax.set_ylabel("Score")
    ax.set_title("Continuous Metrics vs Lead Time")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "continuous_metrics_vs_leadtime.png", dpi=160)
    plt.close(fig)

    if cat:
        csi_map = {}
        pod_map = {}
        far_map = {}
        bias_map = {}
        for row in cat:
            thr = float(row["threshold"])
            lt = int(row["leadtime"])
            csi_map.setdefault(thr, {})[lt] = float(row["CSI"])
            pod_map.setdefault(thr, {})[lt] = float(row["POD"])
            far_map.setdefault(thr, {})[lt] = float(row["FAR"])
            bias_map.setdefault(thr, {})[lt] = float(row["BIAS"])

        plot_thresholds = [1.0, 10.0, 20.0]
        fig, ax = plt.subplots(figsize=(7, 4))
        for thr in plot_thresholds:
            vals = [csi_map.get(thr, {}).get(lt, np.nan) for lt in lead]
            if not np.all(np.isnan(vals)):
                ax.plot(lead, vals, marker="o", label=f"CSI@{int(thr)} mm/h")
        ax.set_xlabel("Lead time (min)")
        ax.set_ylabel("CSI")
        ax.set_title("CSI vs Lead Time")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "csi_vs_leadtime.png", dpi=160)
        plt.close(fig)

        fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharex=True)
        for thr in plot_thresholds:
            pod_vals = [pod_map.get(thr, {}).get(lt, np.nan) for lt in lead]
            far_vals = [far_map.get(thr, {}).get(lt, np.nan) for lt in lead]
            bias_vals = [bias_map.get(thr, {}).get(lt, np.nan) for lt in lead]
            if not np.all(np.isnan(pod_vals)):
                axes[0].plot(lead, pod_vals, marker="o", label=f"{int(thr)} mm/h")
            if not np.all(np.isnan(far_vals)):
                axes[1].plot(lead, far_vals, marker="o", label=f"{int(thr)} mm/h")
            if not np.all(np.isnan(bias_vals)):
                axes[2].plot(lead, bias_vals, marker="o", label=f"{int(thr)} mm/h")
        axes[0].set_title("POD")
        axes[1].set_title("FAR")
        axes[2].set_title("BIAS")
        for ax in axes:
            ax.set_xlabel("Lead time (min)")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        fig.suptitle("Categorical Metrics vs Lead Time")
        fig.tight_layout()
        fig.savefig(out_dir / "categorical_metrics_vs_leadtime.png", dpi=160)
        plt.close(fig)

def setup_dataloader(
    module_path: str = "nowcasting.data.dataloader.RadarDataModule",
    **kwargs
) -> torch.utils.data.DataLoader:
    """Load a DataModule from `module_path` and return its validation dataloader."""
    print(f"Setting up dataloader from: {module_path}")
    data_module = load_module(module_path)

    data_module = data_module(
        **kwargs
    )
    data_module.setup()
    print("Dataloader setup complete.")
    return data_module.val_dataloader()

def setup_model(
    module_path: str,
    config: OmegaConf
) -> torch.nn.Module:
    """Instantiate the model class at `module_path` with `config`."""
    print(f"Setting up model from: {module_path}")
    model = load_module(module_path)

    # Remove 'module_path' from the config.model section before passing the config
    # to the model's constructor. This is only needed for training the model.
    if 'module_path' in config.model:
        config.model.pop('module_path')

    model = model(config)
    print("Model setup complete.")
    return model

def main(config: Optional[str] = None, **kwargs):
    """Evaluate from a YAML config. Extra Fire kwargs override config keys."""
    print("Starting main evaluation process.")
    
    config = OmegaConf.load(config) if (config is not None) else {}
    config.update(kwargs)

    dataloader = setup_dataloader(**config.dataloader)

    model = setup_model(
        module_path=config.eval.pop('model_path'),
        config=config
    )
    
    evaluation(model, dataloader, config, **config.eval)

    print("Main evaluation process finished.")

if __name__ == "__main__":
    Fire(main)