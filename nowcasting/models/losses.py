import torch
import torch.nn as nn
from torch.nn import functional as F

class BalancedLoss(nn.Module):
    def __init__(self,
                 normalization="zscore",
                 min_val=0,
                 max_val=55,
                 max_weight_r=10.0,
                 weight_intensity=1.0,
                 mean_log_rain=0.03019706713265408,
                 std_log_rain=0.5392297631902654,
                 extended=False):
        """
        Initializes the BalancedLoss.
        Args:
            normalization (str): One of ['minmax', 'zscore']
            min_val (float): Minimum dBZ value (corresponding to normalized 0). Default: 0.
            max_val (float): Maximum dBZ value (corresponding to normalized 1). Default: 55.
            mean_log_rain (float): Mean of log rain rate. Default: 0.03019706713265408.
            mean_log_rain (float): Standard Diviation of log rain rate. Default: 0.5392297631902654.
            min_weight (float): Minimum rain rate (mm/h) for clamping the raw weights. Default: 1.0.
            max_weight (float): Maximum rain rate (mm/h) for clamping the raw weights. Default: 10.0.
            weight_intensity (float): A scalar to control the influence of the calculated weights.
                                      0.0 means pure MSE, 1.0 means full weighting. Default: 1.0.
        """
        super(BalancedLoss, self).__init__()
        assert normalization in ['minmax', 'zscore'], "normalization must be 'minmax' or 'zscore'"
        self.normalization = normalization

        self.register_buffer("min_val_dbz", torch.tensor(float(min_val), dtype=torch.float32))
        self.register_buffer("max_val_dbz", torch.tensor(float(max_val), dtype=torch.float32))
        self.register_buffer("mean_log_rain", torch.tensor(float(mean_log_rain), dtype=torch.float32))
        self.register_buffer("std_log_rain", torch.tensor(float(std_log_rain), dtype=torch.float32))
        
        self.register_buffer("max_weight_r", torch.tensor(float(max_weight_r), dtype=torch.float32))
        self.register_buffer("weight_intensity", torch.tensor(float(weight_intensity), dtype=torch.float32))
        self.extended = extended

    def forward(self, pred, target):
        """
        Computes the weighted Mean Squared Error loss.
        Weights are derived from the ground truth target's rain rate, and their influence
        is scaled by `weight_intensity`. 
        Pred and target should both be in normalized form (either minmax or zscore).

        Args:
            pred (torch.Tensor): Model's prediction, normalized.
            target (torch.Tensor): Ground truth target, normalized.

        Returns:
            torch.Tensor: Scalar loss value.
        """

        """
        pred and target should both be in normalized form (either minmax or zscore).
        """
        if self.normalization == 'minmax':
            # Denormalize from [0, 1] to dBZ
            range_dbz = self.max_val_dbz - self.min_val_dbz
            target_dBZ = target * range_dbz + self.min_val_dbz

            # Convert to rain rate
            power_target = target_dBZ / 10.0
            base_for_r_calc = (torch.pow(10.0, power_target) - 1.0) / 200.0
            r_from_target = torch.pow(torch.relu(base_for_r_calc), 5.0 / 8.0)

        elif self.normalization == 'zscore':
            # Denormalize from z-score to log10(rain)
            target_log10_r = target * self.std_log_rain + self.mean_log_rain

            # Convert to rain rate
            r_from_target = torch.pow(10.0, target_log10_r)

        # Clamp and calculate effective weights
        raw_weights = torch.clamp(r_from_target, max=self.max_weight_r) + 1.0
        effective_weights = 1.0 + (raw_weights - 1.0) * self.weight_intensity

        # Compute error (in normalized space, regardless of type)
        mse = (pred - target) ** 2
        weighted_mse = effective_weights * mse

        if self.extended:
            mae = torch.abs(pred - target)
            weighted_mae = effective_weights * mae
            return torch.mean(weighted_mse) + torch.mean(weighted_mae)

        return weighted_mse.mean()
    

class SoftCSILoss(nn.Module):
    """Differentiable approximation of Critical Success Index (CSI).

    Uses sigmoid soft-thresholding to make CSI differentiable.
    Operates in mm/h space — converts from z-score normalized input internally.

    The loss is 1 - CSI, so minimizing it maximizes CSI.

    Parameters
    ----------
    thresholds : list of float
        Precipitation thresholds in mm/h to optimize CSI for.
    sharpness : float
        Steepness of the sigmoid approximation. Higher = closer to hard
        threshold but noisier gradients. 5-20 works well.
    mean_log_rain : float
        Mean of log10(rain) for z-score denormalization.
    std_log_rain : float
        Std of log10(rain) for z-score denormalization.
    """

    def __init__(self, thresholds=(5.0, 10.0), sharpness=10.0,
                 mean_log_rain=0.03019706713265408,
                 std_log_rain=0.5392297631902654,
                 # Accept and ignore extra kwargs from config
                 **kwargs):
        super().__init__()
        self.register_buffer("thresholds", torch.tensor(thresholds, dtype=torch.float32))
        self.sharpness = sharpness
        self.mean_log_rain = mean_log_rain
        self.std_log_rain = std_log_rain

    def _to_mmh(self, x):
        """Convert z-score normalized log-rain to mm/h."""
        x = x * self.std_log_rain + self.mean_log_rain
        return torch.pow(10.0, x)

    def forward(self, pred, target):
        """
        Parameters
        ----------
        pred : torch.Tensor
            Predictions in z-score normalized space.
        target : torch.Tensor
            Targets in z-score normalized space.

        Returns
        -------
        torch.Tensor
            Scalar loss: mean(1 - soft_CSI) across thresholds.
        """
        pred_mmh = self._to_mmh(pred)
        target_mmh = self._to_mmh(target)

        total_loss = 0.0
        for threshold in self.thresholds:
            pred_soft = torch.sigmoid(self.sharpness * (pred_mmh - threshold))
            target_soft = torch.sigmoid(self.sharpness * (target_mmh - threshold))

            tp = (pred_soft * target_soft).sum()
            fp = (pred_soft * (1 - target_soft)).sum()
            fn = ((1 - pred_soft) * target_soft).sum()

            csi = tp / (tp + fp + fn + 1e-8)
            total_loss += (1.0 - csi)

        return total_loss / len(self.thresholds)


class SpectralLoss(nn.Module):
    """Penalizes differences in the 2D spatial power spectrum.

    Computes the 2D FFT of prediction and target, then compares their
    log-amplitude spectra. This forces the model to produce predictions
    with the correct spatial frequency content — sharp edges and fine
    structure rather than smooth blobs.

    Operates on each (B, T) frame independently.

    Parameters
    ----------
    log_scale : bool
        If True, compare log(1 + |FFT|) instead of |FFT|.
        Reduces dominance of low frequencies. Default True.
    """

    def __init__(self, log_scale=True, **kwargs):
        super().__init__()
        self.log_scale = log_scale

    def forward(self, pred, target):
        """
        Parameters
        ----------
        pred, target : torch.Tensor
            Shape (..., H, W) — spatial dims are the last two.

        Returns
        -------
        torch.Tensor
            Scalar loss.
        """
        # 2D FFT over spatial dims, shift zero-freq to center
        pred_fft = torch.fft.fft2(pred, norm='ortho')
        target_fft = torch.fft.fft2(target, norm='ortho')

        pred_amp = torch.abs(pred_fft)
        target_amp = torch.abs(target_fft)

        if self.log_scale:
            pred_amp = torch.log1p(pred_amp)
            target_amp = torch.log1p(target_amp)

        return F.l1_loss(pred_amp, target_amp)


class FACLLoss(nn.Module):
    """Fourier Amplitude and Correlation Loss (Gong et al., NeurIPS 2024).

    Two complementary spectral losses:
    - FAL: Matches Fourier amplitude spectra (translation-invariant)
    - FCL: Normalized Fourier correlation capturing phase/structure

    Training uses curriculum: starts with FCL (structure), gradually
    increases FAL probability (sharpness).

    Parameters
    ----------
    fal_weight : float
        Weight for Fourier Amplitude Loss.
    fcl_weight : float
        Weight for Fourier Correlation Loss.
    curriculum : bool
        If True, use curriculum alternation (FCL first, then FAL).
    curriculum_epochs : int
        Number of epochs over which to transition from FCL to FAL.
    """

    def __init__(self, fal_weight=1.0, fcl_weight=1.0,
                 curriculum=False, curriculum_epochs=20, **kwargs):
        super().__init__()
        self.fal_weight = fal_weight
        self.fcl_weight = fcl_weight
        self.curriculum = curriculum
        self.curriculum_epochs = curriculum_epochs
        self._current_epoch = 0

    def set_epoch(self, epoch):
        """Call this from training_step or on_train_epoch_start."""
        self._current_epoch = epoch

    def _fal(self, pred_fft, target_fft):
        """Fourier Amplitude Loss: L2 between amplitude spectra."""
        pred_amp = torch.abs(pred_fft)
        target_amp = torch.abs(target_fft)
        return F.mse_loss(pred_amp, target_amp)

    def _fcl(self, pred_fft, target_fft):
        """Fourier Correlation Loss: 1 - normalized cross-correlation."""
        cross = (pred_fft * target_fft.conj()).real.sum()
        pred_power = (torch.abs(pred_fft) ** 2).sum()
        target_power = (torch.abs(target_fft) ** 2).sum()
        correlation = cross / (torch.sqrt(pred_power * target_power) + 1e-8)
        return 1.0 - correlation

    def forward(self, pred, target):
        """
        Parameters
        ----------
        pred, target : torch.Tensor
            Shape (..., H, W) — spatial dims are the last two.
        """
        pred_fft = torch.fft.fft2(pred, norm='ortho')
        target_fft = torch.fft.fft2(target, norm='ortho')

        fal = self._fal(pred_fft, target_fft)
        fcl = self._fcl(pred_fft, target_fft)

        if self.curriculum:
            # Linear transition: start with 100% FCL, end with 100% FAL
            progress = min(self._current_epoch / max(self.curriculum_epochs, 1), 1.0)
            w_fal = progress
            w_fcl = 1.0 - progress
            return w_fal * self.fal_weight * fal + w_fcl * self.fcl_weight * fcl

        return self.fal_weight * fal + self.fcl_weight * fcl


class ExceedanceBCELoss(nn.Module):
    """Binary Cross-Entropy loss for precipitation threshold exceedance.

    Instead of regressing rain rate, the model outputs C exceedance
    probabilities P(R >= tau_c) for C thresholds. Each threshold is
    treated as an independent binary classification problem.

    Inspired by RainPro-8 (Andrychowicz et al., 2025), simplified to
    independent BCE without ordinal consistency constraint.

    The model output should have C channels (one per threshold) in the
    last dimension, passed through sigmoid to get probabilities.

    Parameters
    ----------
    thresholds_mmh : list of float
        Precipitation thresholds in mm/h.
    mean_log_rain : float
        Mean of log10(rain) for z-score denormalization.
    std_log_rain : float
        Std of log10(rain) for z-score denormalization.
    pos_weight_max : float
        Maximum positive class weight for rare thresholds. Set to 0 to
        disable reweighting. Higher values upweight rare extreme events.
    """

    def __init__(self, thresholds_mmh=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0),
                 mean_log_rain=0.03019706713265408,
                 std_log_rain=0.5392297631902654,
                 pos_weight_max=0.0,
                 **kwargs):
        super().__init__()
        self.mean_log_rain = mean_log_rain
        self.std_log_rain = std_log_rain
        self.register_buffer(
            "thresholds_mmh",
            torch.tensor(thresholds_mmh, dtype=torch.float32)
        )
        self.n_thresholds = len(thresholds_mmh)
        self.pos_weight_max = pos_weight_max

        # Precompute thresholds in normalized z-score log space
        # z = (log10(R) - mean) / std
        thresholds_zscore = [(torch.log10(torch.tensor(t)) - mean_log_rain) / std_log_rain
                             for t in thresholds_mmh]
        self.register_buffer(
            "thresholds_z",
            torch.stack(thresholds_zscore)
        )

    def forward(self, pred, target):
        """
        Parameters
        ----------
        pred : torch.Tensor
            Model output with shape (..., C) where C = n_thresholds.
            Raw logits (pre-sigmoid).
        target : torch.Tensor
            Ground truth rain rate in z-score normalized log space,
            shape (..., 1) or (...). Will be broadcast to create
            binary exceedance labels for each threshold.

        Returns
        -------
        torch.Tensor
            Scalar loss.
        """
        # Squeeze target if it has a trailing dim of 1
        if target.dim() == pred.dim() and target.shape[-1] == 1:
            target = target.squeeze(-1)

        # Create binary exceedance labels: y_c = 1 if target >= threshold_c
        # target shape: (...), thresholds shape: (C,)
        # Expand for broadcasting
        target_expanded = target.unsqueeze(-1)  # (..., 1)
        thresholds = self.thresholds_z.view(*([1] * (target.dim())), -1)  # (1,...,1, C)
        labels = (target_expanded >= thresholds).float()  # (..., C)

        # BCE with logits (numerically stable)
        if self.pos_weight_max > 0:
            # Compute positive class frequency per threshold for this batch
            pos_freq = labels.mean(dim=tuple(range(labels.dim() - 1)))  # (C,)
            pos_weight = torch.clamp(
                (1 - pos_freq) / (pos_freq + 1e-8),
                max=self.pos_weight_max
            )
            loss = F.binary_cross_entropy_with_logits(
                pred, labels, pos_weight=pos_weight
            )
        else:
            loss = F.binary_cross_entropy_with_logits(pred, labels)

        return loss


class CombinedLoss(nn.Module):
    """Combines BalancedLoss with optional SoftCSILoss and SpectralLoss.

    total_loss = balanced + csi_weight * soft_csi + spectral_weight * spectral

    Parameters
    ----------
    csi_weight : float
        Weight for the soft CSI loss. Set to 0 to disable.
    csi_thresholds : list of float
        mm/h thresholds for CSI optimization.
    csi_sharpness : float
        Sigmoid sharpness for soft thresholding.
    spectral_weight : float
        Weight for the spectral loss. Set to 0 to disable.
    spectral_log_scale : bool
        Use log amplitude spectrum (recommended).

    All other parameters are forwarded to BalancedLoss.
    """

    def __init__(self, csi_weight=0.5, csi_thresholds=(5.0, 10.0),
                 csi_sharpness=10.0,
                 spectral_weight=0.0, spectral_log_scale=True,
                 facl_weight=0.0, facl_fal_weight=1.0, facl_fcl_weight=1.0,
                 facl_curriculum=False, facl_curriculum_epochs=20,
                 **balanced_kwargs):
        super().__init__()
        self.balanced_loss = BalancedLoss(**balanced_kwargs)

        self.csi_weight = csi_weight
        if csi_weight > 0:
            self.csi_loss = SoftCSILoss(
                thresholds=csi_thresholds,
                sharpness=csi_sharpness,
                mean_log_rain=balanced_kwargs.get('mean_log_rain', 0.03019706713265408),
                std_log_rain=balanced_kwargs.get('std_log_rain', 0.5392297631902654),
            )

        self.spectral_weight = spectral_weight
        if spectral_weight > 0:
            self.spectral_loss = SpectralLoss(log_scale=spectral_log_scale)

        self.facl_weight = facl_weight
        if facl_weight > 0:
            self.facl_loss = FACLLoss(
                fal_weight=facl_fal_weight,
                fcl_weight=facl_fcl_weight,
                curriculum=facl_curriculum,
                curriculum_epochs=facl_curriculum_epochs,
            )

    def forward(self, pred, target):
        loss = self.balanced_loss(pred, target)

        if self.csi_weight > 0:
            loss = loss + self.csi_weight * self.csi_loss(pred, target)

        if self.spectral_weight > 0:
            loss = loss + self.spectral_weight * self.spectral_loss(pred, target)

        if self.facl_weight > 0:
            loss = loss + self.facl_weight * self.facl_loss(pred, target)

        return loss


class MultiSourceLoss(nn.Module):
    def __init__(self, 
                 balanced_channels, 
                 mse_channels, 
                 channel_wegihts,
                 min_val=0, 
                 max_val=55, 
                 mean_log_rain=0.03019706713265408,
                 std_log_rain=0.5392297631902654,
                 max_weight_r=10.0, 
                 weight_intensity=1.0, 
                 extended=False):
        """
        Args:
            balanced_channels (list of int): Indices of channels to apply BalancedLoss.
            mse_channels (list of int): Indices of channels to apply plain MSE loss.
            weights (list of float): Weight for each channel (same length as balanced + mse).
            Other args are forwarded to BalancedLoss.
        """
        super(MultiSourceLoss, self).__init__()

        assert len(channel_wegihts) == len(balanced_channels) + len(mse_channels), \
            "weights length must match total number of channels"

        self.balanced_channels = balanced_channels
        self.mse_channels = mse_channels
        self.channel_wegihts = channel_wegihts

        self.balanced_loss = BalancedLoss(
            min_val=min_val,
            max_val=max_val,
            mean_log_rain=mean_log_rain,
            std_log_rain=std_log_rain,
            max_weight_r=max_weight_r,
            weight_intensity=weight_intensity,
            extended=extended
        )

    def forward(self, pred, target):
        """
        Args:
            pred (Tensor): (B, T, H, W, C) prediction tensor
            target (Tensor): (B, T, H, W, C) target tensor
        Returns:
            Scalar tensor representing the weighted total loss.
        """
        total_loss = 0.0
        weight_idx = 0

        for c in self.balanced_channels:
            loss = self.balanced_loss(pred[..., c], target[..., c])
            total_loss += self.channel_wegihts[weight_idx] * loss
            weight_idx += 1

        for c in self.mse_channels:
            mse_loss = F.mse_loss(pred[..., c], target[..., c])
            total_loss += self.channel_wegihts[weight_idx] * mse_loss
            weight_idx += 1

        return total_loss
