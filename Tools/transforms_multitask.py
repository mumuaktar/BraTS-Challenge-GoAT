from monai.transforms import (
    Compose, LoadImaged, ScaleIntensityd, RandRotate90d, NormalizeIntensityd,
    RandSpatialCropd, CenterSpatialCropd, RandCoarseDropoutd, ToTensord, Lambda
)
import numpy as np
import torch



# #####################for same spatial location 
class FullMaskGenerator:
    def __init__(self, patch_size=16, mask_ratio=0.4, device="cuda"):
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.device = device

    def __call__(self, img_shape):
        """
        Args:
            img_shape (tuple): (C, H, W, D)

        Returns:
            torch.Tensor: Binary mask of shape (1, C, H, W, D)
        """
        C, H, W, D = img_shape
        ps = self.patch_size

        # Compute padded dimensions (nearest multiple of patch size)
        pad_H = (H + ps - 1) // ps * ps
        pad_W = (W + ps - 1) // ps * ps
        pad_D = (D + ps - 1) // ps * ps

        # Number of patches
        num_patches = (pad_H // ps, pad_W // ps, pad_D // ps)
        total_patches = np.prod(num_patches)
        num_masked = int(self.mask_ratio * total_patches)

        # Create patch mask
        mask_flat = np.hstack([
            np.ones(num_masked, dtype=np.float32),
            np.zeros(total_patches - num_masked, dtype=np.float32)
        ])
        np.random.shuffle(mask_flat)
        mask_patches = mask_flat.reshape(num_patches)

        # Expand to full padded shape
        mask_full = np.kron(
            mask_patches,
            np.ones((ps, ps, ps), dtype=np.float32)
        )

        # Crop back to original size
        mask_full = mask_full[:H, :W, :D]

        # Convert to tensor [1, C, H, W, D]
        mask_tensor = torch.tensor(mask_full, dtype=torch.float32, device=self.device)
        mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)     # [1, 1, H, W, D]
        mask_tensor = mask_tensor.expand(1, C, -1, -1, -1)      # [1, C, H, W, D]
        return mask_tensor






    
from monai.transforms import MapTransform


class ConvertToMultiChannelBasedOnCustomBratsClassesd(MapTransform):
    """
    Converts label values to multi-channel format for BraTS-like task.
    Your dataset label IDs:
    - 1: necrosis/NCR
    - 2: edema
    - 3: enhancing tumor (ET)

    Channels:
    - Channel 0: Tumor Core (TC) = 1 + 3
    - Channel 1: Whole Tumor (WT) = 1 + 2 + 3
    - Channel 2: Enhancing Tumor (ET) = 3
    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            seg = d[key]  # (C, H, W, D) or (H, W, D)
            
            if isinstance(seg, torch.Tensor):
                seg = seg.numpy()

            # make sure we're working with 3D (no extra channel dim)
            if seg.ndim == 4 and seg.shape[0] == 1:
                seg = np.squeeze(seg, axis=0)

            tc = np.logical_or(seg == 1, seg == 3)   # Tumor Core
            wt = np.logical_or(tc, seg == 2)         # Whole Tumor
            et = seg == 3                             # Enhancing Tumor

            multi_channel = np.stack([tc, wt, et], axis=0).astype(np.float32)  # (3, H, W, D)
            d[key] = multi_channel
        return d
# For training (includes segmentation if available)
def print_shape(d):
    for k, v in d.items():
        print(f"{k}: {v.shape}")
    return d





train_transforms = Compose([
    LoadImaged(keys=["img", "groundtruth", "seg"], allow_missing_keys=True, ensure_channel_first=True),
    ConvertToMultiChannelBasedOnCustomBratsClassesd(keys="seg", allow_missing_keys=True),
    NormalizeIntensityd(keys="img", nonzero=True, channel_wise=True),
    # ScaleIntensityd(keys=["img", "groundtruth"], minv=0.0, maxv=1.0),
    RandRotate90d(keys=["img", "groundtruth", "seg"], prob=0.5, allow_missing_keys=True),
    RandSpatialCropd(keys=["img", "groundtruth", "seg"], roi_size=(96,96,96), random_center=True, allow_missing_keys=True),
    ToTensord(keys=["img", "groundtruth", "seg"], allow_missing_keys=True),  # include is_dummy
])    





val_transforms = Compose([
    LoadImaged(keys=["img", "groundtruth", "seg"], ensure_channel_first=True,allow_missing_keys=True),
    ConvertToMultiChannelBasedOnCustomBratsClassesd(keys="seg",allow_missing_keys=True),
    NormalizeIntensityd(keys="img", nonzero=True, channel_wise=True),
    ToTensord(keys=["img", "groundtruth", "seg"],allow_missing_keys=True),
])




test_transforms = Compose([
    LoadImaged(keys=["img"], ensure_channel_first=True,allow_missing_keys=True),
    # ConvertToMultiChannelBasedOnCustomBratsClassesd(keys="seg",allow_missing_keys=True),
    NormalizeIntensityd(keys="img", nonzero=True, channel_wise=True),
    ToTensord(keys=["img"],allow_missing_keys=True),
])


