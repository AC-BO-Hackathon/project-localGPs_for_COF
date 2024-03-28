# This file contaings surrogate model inputs
import numpy as np     
import os
import json
import math

# Torch specific module imports
import torch
import gpytorch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torch.nn import functional as F

# botorch specific modules
from botorch.fit import fit_gpytorch_model
from botorch.models.gpytorch import GPyTorchModel

# Plotting libraries
import matplotlib as mpl
import matplotlib.pyplot as plt

# User defined python classes and files
import sys

import utils_dataset as utilsd
import input_class 

np.random.seed(0)
torch.manual_seed(0)

# General inputs
run_folder = '/Users/nikhilthota/Desktop/lab/projects/SPIRAL/codes_and_datasets/T-NIKHIL/project-sparse-gp-for-materials-discovery'  # Folder where code is run and input json exist
num_run = 3
test_size = 0.01
output_folder = run_folder+'/bo_output/' # Folder where all outputs are stored
output_folder = output_folder+'test'
verbose = True
deep_verbose = False

# Reading and data processing inputs
add_target_noise = False
standardize_data = True

# Feature selection inputs
test_size_fs = 0.1
select_features_otherModels = False

# BO inputs
n_trials = 5
n_update = 1000
GP_0_BO = True
GP_L_BO = True
GP_NN_BO = False
random_seed = 'iteration'
maximization = True
new_values_predict_from_model = False
n_batch_perTrial = 100

# Surrogate training boolean inputs
train_GP = True

# GP Model parameters
kernel = 'Matern'
learning_rate_gp0 = 0.01
epochs_GP0 = 500
