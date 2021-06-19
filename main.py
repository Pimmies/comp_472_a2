import string

from Model import Model
from Tester import Tester
from scrapUtils import *

#################################################################################################
# 1.1 BUILDING A TRAINING SET AND TESTING SET
#createDataSets()

#################################################################################################
# 1.2 Extract the data and build the model
# tokenize the training set, cleanup and calculate probabilities

training_pos_path = './training_set/training_positive.txt'
training_neg_path = './training_set/training_negative.txt'
testing_pos_path = './testing_set/testing_positive.txt'
testing_neg_path = './testing_set/testing_negative.txt'

# smoothing value
SMOOTHING_VAL = 1

# Training
basic_model = Model(training_pos_path, training_neg_path, SMOOTHING_VAL)

# Testing
tester = Tester(testing_pos_path, testing_neg_path, basic_model)
