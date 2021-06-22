import string

from Model import Model
from Tester import Tester
import matplotlib.pyplot as plt
import numpy as np

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

f1 = plt.figure()
f2 = plt.figure()
ax1 = f1.add_subplot(111)
ax2 = f2.add_subplot(111)

# Training and Testing for Frequency Removals
freq_model = Model(training_pos_path, training_neg_path, SMOOTHING_VAL)
tester = Tester(testing_pos_path, testing_neg_path, freq_model)
freq_testers = [tester]
vocab_sizes = [freq_model.getVocabularySize()]

freq_args = ['=1', '<=10', '<=20', 'top5%', 'top10%', 'top20%']
for arg in freq_args:
    print("Testing for removing frequency " + arg + ":")
    freq_model.removeWordsByFrequency(arg)
    tester = Tester(testing_pos_path, testing_neg_path, freq_model)
    freq_testers.append(tester)
    vocab_sizes.append(freq_model.getVocabularySize())

freq_model.writeToRemoveFile("frequency-remove")

# Plotting for Frequency Removals
x = np.array(vocab_sizes)
y = np.array([tester.getFMeasure(1) for tester in freq_testers])
ax1.plot(x, y)
ax1.title.set_text('Task 2.1 - Frequency Removal')
ax1.set_xlabel('Vocabulary Size')
ax1.set_ylabel('F Measure')
ax1.set_ylim([0, 1])


s_testers = []
smoothing_values = np.linspace(0.6, 2, 8)
for val in smoothing_values:
    print("Testing for model with smoothing value " + str(val) + ":")
    s_model = Model(training_pos_path, training_neg_path, val)
    tester = Tester(testing_pos_path, testing_neg_path, s_model)
    s_testers.append(tester)
y = np.array([tester.getFMeasure(1) for tester in s_testers])
ax2.plot(smoothing_values, y)
ax2.title.set_text('Task 2.2 - Smoothing Values')
ax2.set_xlabel('Smoothing Values')
ax2.set_ylabel('F Measure')
ax2.set_ylim([0.65, 0.9])


plt.show()

# Testing
# tester = Tester(testing_pos_path, testing_neg_path, basic_model)
