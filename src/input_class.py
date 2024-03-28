import sklearn
import numpy as np     
import csv 
import copy 
import random 
import pandas as pd
import pickle
import json  
import openpyxl
import itertools

# User defined files and classes
import feature_selection_methods as feature_selection
import utils_dataset as utilsd

# Plotting libraries
import matplotlib as mpl
import matplotlib.pyplot as plt

# Tick parameters
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.rcParams['xtick.major.size'] = 5
plt.rcParams['xtick.major.width'] = 1
plt.rcParams['xtick.minor.size'] = 5
plt.rcParams['xtick.minor.width'] = 1
plt.rcParams['ytick.major.size'] = 5
plt.rcParams['ytick.major.width'] = 1
plt.rcParams['ytick.minor.size'] = 5
plt.rcParams['ytick.minor.width'] = 1

plt.rcParams['axes.labelsize'] = 20
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['legend.fontsize'] = 15


class inputs:
    def __init__(self,input_type='COF',input_path='.',input_file='properties.csv'):
        self.input_type = input_type
        self.input_path = input_path
        self.input_file = input_file
        self.filename   = self.input_path + self.input_file

    def read_inputs(self):
        '''
        This function reads the dataset from the COF paper: https://pubs.acs.org/doi/10.1021/acs.chemmater.8b01425
        input_type='COF',
        input_path='.',
        input_file='properties.csv'
        '''     
        data = pd.read_csv(self.filename)
        descriptors = ['dimensions', 'bond type', 'void fraction [widom]', 'supercell volume [A^3]', 'density [kg/m^3]', 
                       'heat desorption high P [kJ/mol]','absolute methane uptake high P [molec/unit cell]', 
                       'absolute methane uptake high P [mol/kg]', 'excess methane uptake high P [molec/unit cell]',
                       'excess methane uptake high P [mol/kg]', 'heat desorption low P [kJ/mol]', 
                       'absolute methane uptake low P [molec/unit cell]', 
                       'absolute methane uptake low P [mol/kg]', 
                       'excess methane uptake low P [molec/unit cell]', 
                       'excess methane uptake low P [mol/kg]', 'surface area [m^2/g]', 'linkerA', 'linkerB', 'net', 
                       'cell_a [A]', 'cell_b [A]', 'cell_c [A]', 'alpha [deg]', 'beta [deg]', 'gamma [deg]', 
                       'num carbon', 'num fluorine', 'num hydrogen', 'num nitrogen', 'num oxygen', 'num sulfur', 
                       'num silicon', 'vertices', 'edges', 'genus', 'largest included sphere diameter [A]', 
                       'largest free sphere diameter [A]', 'largest included sphere along free sphere path diameter [A]',
                       'absolute methane uptake high P [v STP/v]', 'absolute methane uptake low P [v STP/v]]']
        XX = pd.DataFrame(data, columns=descriptors)
        target = copy.deepcopy(data['deliverable capacity [v STP/v]'].to_numpy())
        YY = target.reshape(-1,1)

        return XX, YY, descriptors


if __name__=="__main__":
    
    print('Reading inputs')
    