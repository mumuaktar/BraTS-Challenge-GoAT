import monai
import torch
# import wandb
import os
import pandas as pd
import numpy as np
import nibabel as nib
import re
from loss import *
from monai.metrics import SSIMMetric
from load_data_multitask import *
from monai.data import decollate_batch
import nibabel as nib
from monai.transforms import AsDiscrete, Compose, EnsureType, Activations
from transforms_multitask import *
from monai.metrics import DiceMetric
from monai.utils.enums import MetricReduction
from functools import partial
from monai.inferers import sliding_window_inference
from functools import partial
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# wandb.login(key='1956d1a690ce35746295823d0b579497458c402e')
from monai.losses import DiceLoss
import torch.nn.functional as F
from monai.losses import FocalLoss

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


def train(train_loader, val_loader, model, optimizer, scheduler, max_epochs, directory_name, start_epoch=1):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)
    model.train()
    results_dir='/home/mumu.aktar/tumor_seg1/multitask_val_output1'

    criterion = DiceLoss(to_onehot_y=False, sigmoid=True)
    class_weights = torch.tensor([0.4, 0.2, 0.4], device=device).view(1, 3, 1, 1, 1)

    criterion_ce = nn.BCEWithLogitsLoss(pos_weight=class_weights)
    # criterion_focal = FocalLoss(weight=class_weights, sigmoid=True, gamma=2.0)


    post_sigmoid = Activations(sigmoid=True)
    post_pred = AsDiscrete(argmax=False, threshold=0.5)
    dice_metric = DiceMetric(include_background=True, reduction=MetricReduction.MEAN_BATCH, get_not_nans=True)

    checkpoint_path = os.path.join(directory_name, "best_model_multitask_dyn.pth")
    last_model_path = os.path.join(directory_name, "last_model_multitask_dyn.pth")
    training_results_dir = os.path.join(directory_name, "training_multitask_dyn")
    os.makedirs(directory_name, exist_ok=True)
    os.makedirs(training_results_dir, exist_ok=True)

    best_val_loss = float("inf")
    best_dice_score=-1.0
    if os.path.exists(last_model_path):
        checkpoint = torch.load(last_model_path, map_location=device)
        
        ########just to avoid parallel to run on arc
        # state_dict = checkpoint['state_dict']
        # new_state_dict = {}

        # for k, v in state_dict.items():
        #     new_k = k.replace("module.", "") if k.startswith("module.") else k
        #     new_state_dict[new_k] = v

        # # Load into model (not wrapped in DataParallel)
        # model.load_state_dict(new_state_dict)
        
        ###########################################3
     
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        # best_val_loss = checkpoint.get('best_val_loss', float("inf"))
        best_dice_score=checkpoint.get('best_dice_score',-1)
        start_epoch = checkpoint.get('epoch', 1) + 1
        print(f"Last model loaded. Resuming training from epoch: {start_epoch}")

    model_inferer = partial(
        sliding_window_inference,
        roi_size=[96, 96, 96],
        sw_batch_size=2,
        predictor=lambda x: model(x)[0],
        overlap=0.7,
    )
    model = nn.DataParallel(model)
    model=model.to(device)

    def compute_binary_dice(pred, gt, epsilon=1e-5):
        pred = pred.astype(np.uint8)
        gt = gt.astype(np.uint8)
        intersection = (pred & gt).sum()
        return (2.0 * intersection) / (pred.sum() + gt.sum() + epsilon)
    
    def get_alpha_beta(epoch, seg_start=50, max_epoch=200):
        if epoch < seg_start:
            return 0.0, 1.0
        progress = min((epoch - seg_start) / (max_epoch - seg_start), 1.0)
        beta = 0.3 * (1.0 - progress)  # recon decays to 0
        alpha = 1.0 - beta             # seg increases to 1
        return alpha, beta


    for epoch in range(start_epoch, max_epochs + 1):
        print(f"\n🔁 Epoch {epoch}")
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            img = batch["img"].to(device)
            seg = batch.get("seg", None)
            groundtruth = img

            if seg is not None:
                seg = seg.to(device)

            B, C, H, W, D = img.shape
            
            if epoch<50:
                mask_generator = FullMaskGenerator(patch_size=16, mask_ratio=0.4, device=img.device)
                with torch.no_grad():
                    mask = torch.stack([mask_generator((C, H, W, D)).squeeze(0) for _ in range(B)], dim=0)
                    
            elif epoch<100:
                mask_generator = FullMaskGenerator(patch_size=16, mask_ratio=0.25, device=img.device)
                with torch.no_grad():
                    mask = torch.stack([mask_generator((C, H, W, D)).squeeze(0) for _ in range(B)], dim=0)
            
            elif epoch<150:
                mask_generator = FullMaskGenerator(patch_size=16, mask_ratio=0.10, device=img.device)
                with torch.no_grad():
                    mask = torch.stack([mask_generator((C, H, W, D)).squeeze(0) for _ in range(B)], dim=0)
                    
            elif epoch<200:
                mask_generator = FullMaskGenerator(patch_size=16, mask_ratio=0.05, device=img.device)
                with torch.no_grad():
                    mask = torch.stack([mask_generator((C, H, W, D)).squeeze(0) for _ in range(B)], dim=0)
            
            
            else:
                mask = torch.zeros_like(img)
            
                # mask_generator = FullMaskGenerator(patch_size=16, mask_ratio=0.4, device=img.device)
                # with torch.no_grad():
                #     mask = torch.stack([mask_generator((C, H, W, D)).squeeze(0) for _ in range(B)], dim=0)

            masked_input = img * (1 - mask)
            optimizer.zero_grad()
            pred_seg, pred_recon = model(masked_input)

            if epoch < 50:
                loss = F.l1_loss(pred_recon, img)
                train_loss += loss.item()
            else:
                # alpha = 1.0
                # beta = 0.2
                alpha,beta=get_alpha_beta(epoch, seg_start=50, max_epoch=200)
                loss = 0.0
                loss_seg = torch.tensor(0.0, device=img.device)
                valid_seg_count = 0
                
                
                if seg is not None:
                    is_dummy = batch["is_dummy"].to(device)
                    for i in range(seg.shape[0]):
                        if not is_dummy[i]:
                            loss_seg += criterion(pred_seg[i].unsqueeze(0), seg[i].unsqueeze(0))
                            loss_seg += criterion_ce(pred_seg[i].unsqueeze(0), seg[i].unsqueeze(0))
                            valid_seg_count += 1


                    if valid_seg_count > 0:
                        loss_seg /= valid_seg_count
                        loss += alpha * loss_seg

                loss_recon = F.l1_loss(pred_recon, img)
                loss += beta * loss_recon
                train_loss += loss.item()
            loss.backward()
            optimizer.step()
            

        train_loss /= len(train_loader)
        print(f"✅ Training Loss: {train_loss:.4f}")

        # ----------------------
        # Validation
        # ----------------------
        model.eval()
        val_loss = 0.0
        dice_scores = []
        import numpy as np

        affine = np.eye(4)
        dice_metric.reset()
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                img = batch["img"].to(device)
                seg = batch.get("seg", None)
                groundtruth = img

                if seg is not None:
                    seg = seg.to(device)

                B, C, H, W, D = img.shape
                mask_generator = FullMaskGenerator(patch_size=16, mask_ratio=0.4, device=img.device)
                mask = torch.stack([mask_generator((C, H, W, D)).squeeze(0) for _ in range(B)], dim=0)
                masked_input = img * (1 - mask)

              
                pred_seg = model_inferer(img)

             
                is_dummy=None
                if seg is not None:
                     is_dummy = batch["is_dummy"].to(device)
                if seg is not None and (~is_dummy).sum() > 0:  # if there are valid samples
                        val_output_convert = [post_pred(post_sigmoid(p)) for p in pred_seg]
                        pred_seg_onehot = [p for p, d in zip(val_output_convert, is_dummy) if not d]
                        seg_onehot = [s for s, d in zip(seg, is_dummy) if not d]
                    
                        # dice_metric.reset()
                        dice_metric(y_pred=pred_seg_onehot, y=seg_onehot)
                        batch_dice, _ = dice_metric.aggregate()
                        dice_scores.append(batch_dice.detach().cpu())
                        
                        # Get prediction and ground truth for this sample
                        valid_indices = [i for i, d in enumerate(is_dummy) if not d]
                        for idx, i in enumerate(valid_indices):
                            subject_id = batch["subject_id"][i]
                            
                            # Paths
                            img_paths = [os.path.join(directory_name, f"{subject_id}_0000.nii.gz")]
                            img_path = img_paths[0]
                            save_filename = f"{subject_id}"
                        
                            save_img_path = os.path.join(results_dir, f"{save_filename}_gt.nii.gz")
                            save_pred_path = os.path.join(results_dir, f"{save_filename}.nii.gz")
                        
                            affine = nib.load(img_path).affine
                            pred_np = pred_seg_onehot[idx].detach().cpu().numpy().astype(np.uint8)
                            seg_np = seg_onehot[idx].detach().cpu().numpy().astype(np.uint8)
          
                            import numpy as np

                            
                            single_channel_pred = convert_to_single_channel(pred_np)
                            single_channel_gt = convert_to_single_channel(seg_np)

                           
                        
                            
                           
                           # Save this single-channel prediction as NIfTI
                       #     nib.save(nib.Nifti1Image(single_channel_pred, affine), save_pred_path)
                        #    nib.save(nib.Nifti1Image(single_channel_gt, affine), save_img_path)
                            
                            
                            
                
    
        # val_loss /= len(val_loader)
        # print(f"📉 Validation Loss: {val_loss:.4f}")
        # Given pred and gt in shape [B, C, H, W, D] or similar:
# Region composition
        if dice_scores:  # i.e., not empty
            all_dice_tensor = torch.stack(dice_scores, dim=0)
            per_class_dice = torch.nanmean(all_dice_tensor, dim=0)
            mean_dice = all_dice_tensor.mean().item()
            print(f"📊 Dice Scores — TC: {per_class_dice[0].item():.4f}, WT: {per_class_dice[1].item():.4f}, ET: {per_class_dice[2].item():.4f}")
            print(f"🌟 Mean Dice: {mean_dice:.4f}")
        else:
            mean_dice = 0.0  # or float("nan") if you want to indicate invalidity

        if mean_dice > best_dice_score:
            best_dice_score = mean_dice
            torch.save({
                'epoch': epoch,
                "state_dict": model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                # "val_loss": val_loss,
                "best_dice_score": best_dice_score
            }, checkpoint_path)
            print("✅ Best model saved based on Dice score.")
        
                
        
       

        # Update learning rate
        

        

        # # Save the best model
        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     torch.save({
        #         'epoch': epoch,
        #         # "state_dict": model.state_dict(),
        #         "state_dict":model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(),
        #         "optimizer": optimizer.state_dict(),
        #         "scheduler": scheduler.state_dict(),
        #         "val_loss": val_loss,
        #         "best_val_loss": best_val_loss
        #     }, checkpoint_path)
        #     print("Best model saved.")

        # Save the last model
        torch.save({
            'epoch': epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            # "val_loss": val_loss,
            # "best_val_loss": best_val_loss,
            "best_dice_score":best_dice_score
        }, last_model_path)

        # Step the scheduler
        scheduler.step()
    print("Training complete.")


# torch.save({
#     'state_dict': model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
# }, save_path)



# def load_last_model(model, optimizer, scheduler, directory_name, reset_lr=None):
#     """
#     Load the last saved model checkpoint to resume training if available.

#     Args:
#         model (torch.nn.Module): Model to load weights into.
#         optimizer (torch.optim.Optimizer): Optimizer to load state.
#         scheduler (torch.optim.lr_scheduler._LRScheduler): Scheduler to load state.
#         directory_name (str): Directory where model checkpoints are stored.
#         reset_lr (float, optional): Optionally reset learning rate to a specified value.

#     Returns:
#         tuple: Updated model, optimizer, scheduler, start_epoch, last_val_loss, and best_val_loss.
#     """
#     # Load the last model weights
#     last_model_path = os.path.join(directory_name, "last_model_enlarged_FL.pt")
#     if os.path.exists(last_model_path):
#         checkpoint = torch.load(last_model_path)
#         model.load_state_dict(checkpoint['state_dict'])
#         optimizer.load_state_dict(checkpoint['optimizer'])
#         scheduler.load_state_dict(checkpoint['scheduler'])
#         start_epoch = checkpoint['epoch'] + 1
#         last_val_loss = checkpoint['val_loss']
#         best_val_loss = checkpoint.get('best_val_loss', float('inf'))  # Get the best val loss from checkpoint if available
#         print(f"Last model loaded. Resuming training from epoch {start_epoch}")
        
#         # Optionally reset the learning rate if a new learning rate is provided
#         if reset_lr is not None:
#             for param_group in optimizer.param_groups:
#                 param_group['lr'] = reset_lr
#             print(f"Learning Rate after resetting: {optimizer.param_groups[0]['lr']}")
        
#         return model, optimizer, scheduler, start_epoch, last_val_loss, best_val_loss
#     else:
#         print("Last model weights not found. Starting training from scratch.")
#         return model, optimizer, scheduler, 1, float('inf'), float('inf')
