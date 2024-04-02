"This file contains surrogate model inputs"
import os

# Specify number of clusters for the k mean clustering
NUM_CLUSTER = 3
# Where to store the output
RUN_FOLDER = os.path.dirname(os.getcwd())
# Percentage of data that is used for training the GP (1-test_size)
TEST_SIZE = 0.95
# Output folder name
OUT_FOLDER = RUN_FOLDER+'/bo_output/'
# OUT_FOLDER = OUT_FOLDER+'results_with_three_clusters/'
OUT_FOLDER = OUT_FOLDER+'results_with_three_clusters/'
VERBOSE = True
DEEP_VERBOSE = False
# Initial search before epsilon greedy search train GPs in parallel ?
TRAIN_PARALLEL = False

# Reading and data processing inputs
ADD_TARGET_NOISE = False
STANDARDIZE = True

#### Not Implemented Yet. Stay tuned ! ####
# Feature selection inputs
TEST_SIZE_FS = 0.1
select_features_otherModels = False
############################################

# BO inputs
RANDOM_SEED = 'time'
MAXIMIZATION = True
NEW_VALUES_PREDICT_FROM_MODEL = False
# Number of trials to perform BO from scratch
N_TRIALS = 10
# Number of experiments performed in each trial
N_BATCH_PER_TRIAL = 300
# Number of iterations after which the model is updated
N_UPDATE = 10
# Number of times to perform search. 
# If N_SEARCH is set to 1 then only the initial search is performed
# If N_SEARCH is set to > 1 then only the epsilon greedy search is performed
N_SEARCH = 6
# Epsilon greedy search parameter
EPSILON = 0.1
# Select the GP surrogate to train
GP_0_BO = True
#### Not Implemented Yet. Stay tuned ! ####
GP_L_BO = True
GP_NN_BO = False
############################################

# Surrogate training boolean inputs
TRAIN_GP = True

# GP Model parameters
KERNEL = 'Matern'
LR_GP0 = 0.01
EPOCHS_GP0 = 500
