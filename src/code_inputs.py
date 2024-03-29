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
OUT_FOLDER = OUT_FOLDER+'test_with_cluster/'
VERBOSE = True
DEEP_VERBOSE = False

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
N_TRIALS = 1
# Number of experiments performed in each trial
N_BATCH_PER_TRIAL = 1
# Number of iterations after which the model is updated
N_UPDATE = 1
# Number of times to perform epsilon greedy search. N_BATCH_PER_TRIAL/N_SEARCH must be an integer
N_SEARCH = 1
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
