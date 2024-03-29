# This file costructs surrogate models for the input datasets
import numpy as np     
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import math

# Torch specific module imports
import torch
import gpytorch

# botorch specific modules
from botorch.fit import fit_gpytorch_model
from botorch.models.gpytorch import GPyTorchModel

# Plotting libraries
import matplotlib as mpl
import matplotlib.pyplot as plt

# User defined python classes and files
import utils_dataset as utilsd
import input_class 
import code_inputs as model_input

np.random.seed(0)
torch.manual_seed(0)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
    
# We will use the simplest form of GP model, exact inference
class ExactGPModel(gpytorch.models.ExactGP,GPyTorchModel):
    _num_outputs = 1  # to inform GPyTorchModel API
    MIN_INFERRED_NOISE_LEVEL = 1e-5
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        if model_input.kernel=='RBF':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        elif model_input.kernel=='Matern':            
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=0.5))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
            
#--------------------------- GP-0 ---------------------------#
def train_surrogate_gp0(X_train,Y_train):
    
    mse_gp0 = 0.0 
    training_iter = model_input.epochs_GP0
    
    # initialize likelihood and model
    likelihood_gp0 = gpytorch.likelihoods.GaussianLikelihood()
    model_gp0 = ExactGPModel(X_train, Y_train, likelihood_gp0) 
    
    # Find optimal model hyperparameters
    model_gp0.train()
    likelihood_gp0.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model_gp0.parameters(), lr=model_input.learning_rate_gp0)  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood_gp0, model_gp0)

    for i in range(training_iter):
        optimizer.zero_grad()        # Zero gradients from previous iteration
        output = model_gp0(X_train)  # Output from model
        loss = -mll(output, Y_train) # Calc loss and backprop gradients         
        loss.backward()
        optimizer.step()
        
    return model_gp0, likelihood_gp0

def predict_surrogates(model, likelihood, X):
    
    # Get into evaluation (predictive posterior) mode
    model.eval()
    likelihood.eval()

    # Make predictions by feeding model through likelihood
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        prediction = model(X)
        prediction = likelihood(model(X))

    observed_mean = prediction.mean
    observed_var = prediction.variance
    observed_covar = prediction.covariance_matrix

    return observed_mean, observed_var
    
    