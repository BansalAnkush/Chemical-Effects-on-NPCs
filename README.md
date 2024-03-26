# **Investigating Effects of Chemicals on Human Neural Cells Through Calcium Imaging and Deep Learning**
This repository contains code for the paper "Investigating Effects of Chemicals on Human Neural Cells Through Calcium Imaging and Deep Learning" by Ray Yueh Ku, Ankush Bansal, et al.

**Overview**
In this study, we investigated the effects of various chemicals on human neural cells using calcium imaging and deep learning techniques. The chemicals studied include VPA, DZM, CAF, and ETOH at low, medium and high concentrations with respective controls.

The key components of this project are:

-**Benchmarking algorithms** against Google AutoML Vision
-**5-fold cross-validation** script for manual splitting of training, validation and test datasets
-**t-SNE visualization** to plot spontaneous and stimulated data of D1 and D2 cell lines for the different chemical conditions

**Repository Structure**
Algorithms_Benchmarking/: Scripts for benchmarking various ML algorithms against Google AutoML Vision
Data_Shuffling/: Contains the 5-fold cross-validation script Random_Shuffle_5_fold_cross_validation.py for manually splitting datasets
tSNE/: Code for generating the t-SNE plot tSNE.py and the outputted visualization
README.md: This file providing an overview of the project

**Dataset**
The calcium imaging dataset used in this study consists of spontaneous and stimulated recordings from D1 and D2 human neural cell lines exposed to low, medium and high concentrations of VPA, DZM, CAF, ETOH and respective controls. The raw data is not included in this repository but available on request to coresponding author.

**Getting Started**
To run the code in this repository:

Clone the repo
Install the required Python packages
Run individual scripts in the Algorithms_Benchmarking/, Data_Shuffling/ and tSNE/ directories as needed


**Citation**
If you use the code or findings from this study, please cite our paper:

Ku, R.Y., Bansal, A., Dutta, D.J., Yamashita, S., Peloquin, J., Vu, D., Shen, Y., Uchida, T., Torii, M. and Hashimoto-Torii, K., 2024. Investigating Effects of Chemicals on Human Neural Cells Through Calcium Imaging and Deep Learning.
