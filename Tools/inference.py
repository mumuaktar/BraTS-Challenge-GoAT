import os
import numpy as np
import pandas as pd
import torch
import nibabel as nib
import wandb
from monai.inferers import SlidingWindowInferer, sliding_window_inference
from torch.nn import CrossEntropyLoss

from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.utils.enums import MetricReduction
from monai.data import decollate_batch
import nibabel as nib
from monai.transforms import AsDiscrete, Compose, EnsureType
from collections import OrderedDict
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.utils.enums import MetricReduction
from monai.metrics import compute_hausdorff_distance
from monai.data import decollate_batch
from monai.losses import DiceLoss
from monai.transforms import AsDiscrete, Compose, EnsureType, Activations
from monai.metrics import DiceMetric
from monai.utils.enums import MetricReduction

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


from monai.transforms import (
    Compose, LoadImaged, NormalizeIntensityd,
    Flipd, Rotate90d, ToTensord
)




# Define tensor-level TTA transforms as lambdas
tta_transforms = [
    lambda x: x,  # identity
    lambda x: torch.flip(x, dims=[2]),  # flip x-axis (assuming C,H,W,D format -> H=2)
    lambda x: torch.flip(x, dims=[3]),  # flip y-axis
    lambda x: torch.flip(x, dims=[4]),  # flip z-axis
    lambda x: torch.rot90(x, k=1, dims=[3,4]),  # rotate 90° around y-z plane
]

def invert_tta(x, t):
    # Inverse operations for each transform, match order with tta_transforms list

    # Identity inverse is identity
    if t == tta_transforms[0]:
        return x
    # flip is its own inverse
    elif t == tta_transforms[1]:
        return torch.flip(x, dims=[2])
    elif t == tta_transforms[2]:
        return torch.flip(x, dims=[3])
    elif t == tta_transforms[3]:
        return torch.flip(x, dims=[4])
    elif t == tta_transforms[4]:
        # inverse rotation is rotating 3 times (270 degrees)
        return torch.rot90(x, k=3, dims=[3,4])
    else:
        return x
def convert_to_single_channel(multi_channel_np: np.ndarray) -> np.ndarray:
    """
    Convert BraTS-style one-hot (3, H, W, D) prediction or GT to single-channel label map:
        0: Background
        1: Tumor Core (TC) [label 1 in GT]
        2: Edema [label 2 in GT]
        3: Enhancing Tumor (ET) [label 3 in GT]

    Assumes:
        Channel 0: TC = 1 + 3
        Channel 1: WT = 1 + 2 + 3
        Channel 2: ET = 3
    """
    assert multi_channel_np.shape[0] == 3, "Expected 3 channels (TC, WT, ET)"
    
    tc = multi_channel_np[0]
    et = multi_channel_np[2]

    output = np.zeros_like(tc, dtype=np.uint8)

    # Priority-based assignment
    output[tc == 1] = 1  # TC gets label 1 (includes necrosis and ET)
    output[(multi_channel_np[1] == 1) & (tc == 0) & (et == 0)] = 2  # Edema only gets label 2
    output[et == 1] = 3  # Enhancing Tumor gets label 3 (overwrites TC if needed)

    return output


def test(test_loader, model, input_dir, results_dir):
    device = next(model.parameters()).device
   
    os.makedirs(results_dir, exist_ok=True)

    
    
    
    model.eval()
    # test_df = pd.read_csv(test_csv)
    from functools import partial
    from monai.inferers import sliding_window_inference
    # roi = (96,96,96)
    model_inferer = partial(
        sliding_window_inference,
        roi_size=[128,128,128],
        sw_batch_size=1,
        predictor=lambda x: model(x)[0],
        overlap=0.7,
    )

    criterion = DiceLoss(to_onehot_y=False, sigmoid=True)
    post_sigmoid = Activations(sigmoid=True)
    post_pred = AsDiscrete(argmax=False, threshold=0.5)
    dice_metric = DiceMetric(include_background=True, reduction=MetricReduction.MEAN_BATCH, get_not_nans=True)
    hausdorff_distances = []  # Store Hausdorff distances for each batch
    
    # dice_metric = DiceMetric(include_background=True, reduction="none", get_not_nans=True)
    total_loss = 0.0
    total_batches = 0
    dice_scores = []
    all_dice=[]
    presence_mask=[]
    # test_result_dir='/home/ai2lab/Downloads/miccai2025/test_result_enlarged_FL_focal_best'
    import glob

    print('entered')
    
    print("Checking samples from val_loader...")

  

    for i, batch in enumerate(test_loader):
        print(f"Batch {i}:")
        if isinstance(batch, dict):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    print(f"  {k}: shape={v.shape} dtype={v.dtype}")
                else:
                    print(f"  {k}: type={type(v)} value={v}")
        else:
            print(f"  {batch}")
        if i >= 2:  # print only first 3 batches
            break
    
    print("Sample check done.")

    
    with torch.no_grad():
       
        for batch_idx, batch in enumerate(test_loader):
            subject_id = batch["subject_id"][0]
          
           
            print('testing:',subject_id)
            img = batch["img"].to(device)
    
            
            # pred_seg = model_inferer(img)
            tta_preds = []
            for tta in tta_transforms:
                img_aug = tta(img)
                pred_aug = model_inferer(img_aug)
                pred_reverted = invert_tta(pred_aug, tta)
                tta_preds.append(pred_reverted)
            
            pred_seg = torch.stack(tta_preds).mean(dim=0)

        
          
            val_outputs_list = [p for p in pred_seg]
 
            val_output_convert = [post_pred(post_sigmoid(val_pred_tensor)) for val_pred_tensor in val_outputs_list]
            pred_seg_onehot =val_output_convert
          
     
            # print("✔️ One-hot shapes:", pred_seg_onehot[0].shape, seg_onehot[0].shape)
            
            
            
            subject_id = batch["subject_id"][0]  # The first element in the batch
            print(subject_id)
            affine_modality = "t1c"
            img_path = os.path.join(input_dir, subject_id, f"{subject_id}-{affine_modality}.nii.gz")

# Now this should succeed if the file exists
            affine = nib.load(img_path).affine

            
            # img_paths = [os.path.join(input_dir, f"{subject_id}_0000.nii.gz"),
            #             os.path.join(input_dir, f"{subject_id}_0001.nii.gz"),  # Modality 2
            #             os.path.join(input_dir, f"{subject_id}_0002.nii.gz"),  # Modality 3
            #             os.path.join(input_dir, f"{subject_id}_0003.nii.gz")  ]
            # img_path=img_paths[0]
            # print(img_path)
            # Save filename based on the subject_id (use modality suffix as well)
            save_filename = f"{subject_id}"
    
            import numpy as np
           
            
            save_pred_path = os.path.join(results_dir, f"{save_filename}.nii.gz")
            
            affine = nib.load(img_path).affine
            # Save the images
            # nib.save(nib.Nifti1Image(img_np, affine), save_img_path)
            # After converting predictions to numpy
            pred_np = pred_seg_onehot[0].detach().cpu().numpy().astype(np.uint8)
         
            
            # Convert to single-channel (with correct label encoding)
            single_channel_pred = convert_to_single_channel(pred_np)
            # single_channel_gt = convert_to_single_channel(seg_np)  # If saving GT
            
            # Save NIfTI
            nib.save(nib.Nifti1Image(single_channel_pred, affine), save_pred_path)

            
            print('saved')
            
  