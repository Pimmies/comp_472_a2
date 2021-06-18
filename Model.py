import os
import string
from sklearn.feature_extraction.text import CountVectorizer
import math


# Converting a txt file of training reviews in to a list of strings in memory
def readTrainingReviewsFromFile(path):
    file = open(path, 'r')
    lines = []
    count = 0
    while True:
        # Get next line from file
        line = file.readline()
        if line != '/\n':
            lines.append(line.rstrip())
        else:
            count = count + 1  # If the line is a /, it means a review is finished
        if not line:  # if line is empty end of file is reached
            break
    file.close()
    return lines, count


# Tokenizing a string, first strip all the punctuation and then return a list of words
def tokenize(txt):
    return txt.translate(str.maketrans('', '', string.punctuation)).split()


class Model:
    """This class is a simple Naive Bayes Model"""
    def __init__(self, training_pos_path, training_neg_path, smooth_val):
        self.smooth_val = smooth_val
        if os.path.exists(training_pos_path) and os.path.exists(training_neg_path):  # - Check if training data exists
            # list of reviews, and review count
            self.pos_reviews, self.pos_reviews_count = readTrainingReviewsFromFile(training_pos_path)
            self.neg_reviews, self.neg_reviews_count = readTrainingReviewsFromFile(training_neg_path)
            # word frequency info
            self.pos_freq = self.getWordInfo(self.pos_reviews)
            self.neg_freq = self.getWordInfo(self.neg_reviews)
            # TODO: Combine the 2 infos into one list and write to file
        else:
            print("Training Path doesn't exist")

    # Process list of reviews and return a list of each word's frequency and probability
    def getWordInfo(self, review_list):
        # Custom tokenizer overrides the default which ignore 1 letter word
        vec = CountVectorizer(tokenizer=lambda txt: tokenize(txt))
        X = vec.fit_transform(review_list)
        word_list = vec.get_feature_names()
        count_list = X.toarray().sum(axis=0)
        prob_list = self.calculateProbabilityList(count_list)
        return list(zip(word_list, count_list, prob_list))

    # Return a list of each word's probability of appearing
    def calculateProbabilityList(self, count_list):
        total_word_count = sum(count_list)
        prob_list = []
        for count in count_list:
            # Calculating the probability of each word showing up in the vocabulary
            # Each probability is smoothed with a a global variable SMOOTHING_VAL
            prob = (count + self.smooth_val) / total_word_count
            prob_list.append(prob)
        return prob_list
