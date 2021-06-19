import os
import string
from sklearn.feature_extraction.text import CountVectorizer
import math
import re


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

# Finds a whole word rather than a substring
def findWholeWord(w):
    return re.compile(r'\b({0})\b'.format(w), flags=re.IGNORECASE).search

class Model:
    """This class is a simple Naive Bayes Model"""
    def __init__(self, training_pos_path, training_neg_path, smooth_val):
        self.smooth_val = smooth_val
        if os.path.exists(training_pos_path) and os.path.exists(training_neg_path):  # - Check if training data exists
            # list of reviews, and review count
            self.pos_reviews, self.pos_reviews_count = readTrainingReviewsFromFile(training_pos_path)
            self.neg_reviews, self.neg_reviews_count = readTrainingReviewsFromFile(training_neg_path)
            # word frequency and conditional probability info
            self.pos_info = self.getWordInfo(self.pos_reviews)
            # DEBUG: print(self.pos_info)
            self.neg_info = self.getWordInfo(self.neg_reviews)
            # print(self.neg_info)
            self.allWordInfo = self.combinePosNegInfo()
            # DEBUG: print(self.allWordInfo)
            # write to file
            self.writeToModelFile("model")
        else:
            print("Training Path doesn't exist")

    # Process list of reviews and return a list of each word's frequency and conditional probability
    def getWordInfo(self, review_list):
        # Custom tokenizer overrides the default which ignore 1 letter word
        vec = CountVectorizer(tokenizer=lambda txt: tokenize(txt))
        X = vec.fit_transform(review_list)
        word_list = vec.get_feature_names()
        count_list = X.toarray().sum(axis=0)
        prob_list = self.calculateProbabilityList(count_list)
        return list(map(list, zip(word_list, count_list, prob_list)))

    # Return a list of each word's conditional probability of appearing
    def calculateProbabilityList(self, count_list):
        total_word_count = sum(count_list)
        prob_list = []
        for count in count_list:
            # Calculating the probability of each word showing up in the vocabulary
            # Each probability is smoothed with a a global variable SMOOTHING_VAL
            prob = math.log10((count + self.smooth_val) / (total_word_count + (self.smooth_val * len(count_list))))
            prob_list.append(prob)
        return prob_list

    # Returns a list combination of positive and negative info [[word, frequency positive, prob positive, frequency negative, prob negative], ...]
    def combinePosNegInfo(self):
        combine = []
        # Add all the positive words and info to list
        for item in self.pos_info:
            combine.append(item)
        # Add all the negative words and info to list
        for item in self.neg_info:
            index = None
            # Make sure the word isn't already in the list
            for word in combine:
                if findWholeWord(item[0])(word[0]):
                    index = combine.index(word)
            if index is not None: # if word is in the list
                # Append to the word the negative freq and prob
                combine[index].append(item[1])
                combine[index].append(item[2])
            else:
                newItem = [item[0], 0.0, 0.0, item[1], item[2]]
                combine.append(newItem)
        # Looks for words that have no value for negative freq and prob
        for item in combine:
            if len(item) == 3:
                item.append(0.0)
                item.append(0.0)
        return combine

    # Writes words, frequency and probability to model file
    def writeToModelFile(self, fileName):
        f = open(fileName + ".txt", "w", encoding='utf-8')
        counter = 1
        for item in self.allWordInfo:
            f.write("No." + str(counter) + " " + item[0] + "\n")
            f.write(str(item[1]) + ", " + str(item[2]) + ", " + str(item[3]) + ", " + str(item[4]) + "\n")
            counter += 1
        f.close()
