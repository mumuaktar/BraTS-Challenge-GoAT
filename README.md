# A Multitask Learning Approach for Segmenting Brain Tumor Sub-regions: Towards Better Generalization
This repository contains the code for multitasking framework proposed to segment brain tumor sub-regions. Heterogeneity in tumor size, location, imaging protocols, and patient demographics leads to significant variability in appear-
ance, making the task highly challenging. In this work, we leverage the Swin UNETR, a transformer-based model designed to capture both local and global dependencies, making
it well-suited for segmenting complex and variable tumor structures. To address the challenge of limited labeled data and enhance generalizability across centers, we employ a multitask learning framework that jointly
performs self-supervised reconstruction and supervised segmentation, enabling robust feature learning through combined task optimization. We evaluated our approach in the BraTS 2025 Challenge dataset, focusing
on the segmentation of three key subregions: whole tumor (WT), tumor core (TC), and enhancing tumor (ET). Our method achieves an average Dice score of 0.72 (0.73,0.76,0.68 for WT, TC and ET respectively) in the
validation set, demonstrating strong performance and robustness under varying clinical conditions. 


Currently, it contains the final docker submission version to the challenge. 

If you find our project useful for your research, please consider citing our paper and codebase with the following: 

Aktar M, Nasser T, Souza R, A Multitask Learning Approach for Segmenting Brain Tumor Subregions: Towards Better Generalization, In BraTS Lighthouse 2025 Challenge: BraTS Generalizability Across Tumors (BraTS-GoAT), MICCAI 2025 (Accepted).
