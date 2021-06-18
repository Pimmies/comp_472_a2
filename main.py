import math
import string

from scrapUtils import *
from sklearn.feature_extraction.text import CountVectorizer

#################################################################################################
# 1.1 BUILDING A TRAINING SET AND TESTING SET
# createDataSets()

#################################################################################################
# 1.2 Extract the data and build the model
# tokenize the training set, cleanup and calculate probabilities

training_pos_path = './training_set/training_positive.txt'
training_neg_path = './training_set/training_negative.txt'
testing_pos_path = './testing_set/testing_positive.txt'
testing_neg_path = './testing_set/testing_negative.txt'

# smoothing value
SMOOTHING_VAL = 1


# Converting a txt file of reviews in to a list of strings in memory
def getReviewsFromFile(path):
    file = open(path, 'r')
    lines = []
    count = 0;
    while True:
        # Get next line from file
        line = file.readline()
        if line != '/\n':
            lines.append(line.rstrip())
        else:
            count = count + 1 # If the line is a /, it means a review is finished
        if not line:  # if line is empty end of file is reached
            break
    file.close()
    return lines, count


# Tokenizing a string, first strip all the punctuation and then return a list of words
def tokenize(txt):
    return txt.translate(str.maketrans('', '', string.punctuation)).split()


# Return a list of each word's probability of appearing
def calculateProbabilityList(count_list):
    total_word_count = sum(count_list)
    prob_list = []
    for count in count_list:
        # Calculating the probability of each word showing up in the vocabulary
        # Each probability is smoothed with a a global variable SMOOTHING_VAL
        prob = (count + SMOOTHING_VAL) / total_word_count
        prob_list.append(prob)
    return prob_list


# Process list of reviews and return a list of each word's frequency and probability
def getWordInfo(review_list):
    # Custom tokenizer overrides the default which ignore 1 letter word
    vec = CountVectorizer(tokenizer=lambda txt: tokenize(txt))
    X = vec.fit_transform(review_list)
    word_list = vec.get_feature_names()
    count_list = X.toarray().sum(axis=0)
    prob_list = calculateProbabilityList(count_list)
    return list(zip(word_list, count_list, prob_list))


if os.path.exists(training_pos_path) and os.path.exists(training_neg_path):  # - Check if training data exists
    # list of reviews
    pos_reviews = getReviewsFromFile(training_pos_path)[0]
    neg_reviews = getReviewsFromFile(training_neg_path)[0]
    # reviews count
    pos_reviews_count = getReviewsFromFile(training_pos_path)[1]
    neg_reviews_count = getReviewsFromFile(training_neg_path)[1]
    # word frequency info
    pos_freq = getWordInfo(pos_reviews)
    neg_freq = getWordInfo(neg_reviews)
    # TODO: Combine the 2 infos into one list and write to file


# Start Testing
if os.path.exists(testing_pos_path) and os.path.exists(testing_neg_path):  # - Check if testing data exists
    pass