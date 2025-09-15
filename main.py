#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  4 01:45:16 2025

@author: mumuaktar
"""

import os
import argparse
import torch
import torch.nn as nn
import nibabel as nib
import numpy as np
from monai.networks.nets import SwinUNETR
from tools.load_data_validation import load_data_validation  
from tools.inference import * 

class MultiTaskSwinUNETR(nn.Module):
    def __init__(self,
                 img_size=(256, 256, 160),
                 in_channels=4,
                 feature_size=48,
                 seg_out_channels=3,
                 recon_out_channels=4,
                 use_checkpoint=True):
        super().__init__()
        self.backbone = SwinUNETR(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=feature_size,
            feature_size=feature_size,
            use_checkpoint=use_checkpoint)
        self.seg_head = nn.Conv3d(feature_size, seg_out_channels, kernel_size=1)
        self.recon_head = nn.Conv3d(feature_size, recon_out_channels, kernel_size=1)

    
    def forward(self, x, recon_input=None):
        seg_features = self.backbone(x)
    
        if recon_input is None or recon_input is x:
            recon_features = seg_features  # reuse features
        else:
            recon_features = self.backbone(recon_input)
    
        seg_output = self.seg_head(seg_features)
        recon_output = self.recon_head(recon_features)
    
        return seg_output, recon_output

def main():

    
    print("Starting main...")
    parser = argparse.ArgumentParser(description="BraTS Inference")
    parser.add_argument("-i", "--input_dir", required=True, help="Path to input data folder")
    parser.add_argument("-o", "--output_dir", required=True, help="Path to save output segmentations")
    args = parser.parse_args()
    print(f"Input dir: {args.input_dir}")
    print(f"Output dir: {args.output_dir}")
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model and load checkpoint
    model = MultiTaskSwinUNETR(in_channels=4, seg_out_channels=3, recon_out_channels=4, feature_size=48)
    model.to(device)
    model.eval()
    
    checkpoint_path = "checkpoints/best_model_multitask_FL2.pth" 
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    print("Checkpoint loaded.")
    os.makedirs(args.output_dir, exist_ok=True)
   


    val_loader = load_data_validation(args.input_dir)
   
    
    
    # Run inference and save predictions
    with torch.no_grad():
        test(val_loader, model, args.input_dir, args.output_dir)  

if __name__ == "__main__":
    # main()
  
    try:
       main()
    except Exception as e:
        print("Error:", e)
        import traceback
        traceback.print_exc()

