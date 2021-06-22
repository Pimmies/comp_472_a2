import os
import string
import time

from sklearn.feature_extraction.text import CountVectorizer
import math
import re
import copy


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
    return txt.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation))).split()


class Model:
    """This class is a simple Naive Bayes Model"""
    def __init__(self, training_pos_path, training_neg_path, smooth_val):
        self.smooth_val = smooth_val
        self.removed_words_list = []
        if os.path.exists(training_pos_path) and os.path.exists(training_neg_path):  # - Check if training data exists
            # list of reviews, and review count
            self.pos_reviews, self.pos_reviews_count = readTrainingReviewsFromFile(training_pos_path)
            self.neg_reviews, self.neg_reviews_count = readTrainingReviewsFromFile(training_neg_path)
            # word frequency and conditional probability info
            self.pos_info = self.initiateWordInfoWithDataset(self.pos_reviews)
            self.neg_info = self.initiateWordInfoWithDataset(self.neg_reviews)
            self.allWordInfo = self.combinePosNegInfo()
        else:
            print("Training Path doesn't exist")

    # Process list of reviews and return a list of each word's frequency and conditional probability
    def initiateWordInfoWithDataset(self, review_list):
        # Custom tokenizer overrides the default which ignore 1 letter word
        vec = CountVectorizer(tokenizer=lambda txt: tokenize(txt), stop_words='english')
        self.removed_words_list = list(vec.get_stop_words())
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

    # Returns the smoothed probability of a non-appearing new word
    # review_type: "positive" or "negative"
    def newWordProbability(self, review_type):
        if review_type in "positive":
            info = self.pos_info
        else:
            info = self.neg_info
        unique_word_count = len(info)
        total_word_count = 0
        for item in info:
            total_word_count += item[1]
        return math.log10(self.smooth_val / (total_word_count + (self.smooth_val * unique_word_count)))

    # Returns a list combination of positive and negative info [[word, frequency positive, prob positive, frequency negative, prob negative], ...]
    def combinePosNegInfo(self):
        combine = []
        new_word_prob_pos = self.newWordProbability("positive")
        new_word_prob_neg = self.newWordProbability("negative")
        # Add all the positive words and info to list
        for item in self.pos_info:
            combine.append(copy.deepcopy(item))
        # Add all the negative words and info to list
        for item in self.neg_info:
            index = None
            for word in combine:
                # Make sure the word isn't already in the list
                if item[0] == word[0]:
                    index = combine.index(word)
                    break
            if index is not None: # if word is in the list
                # Append to the word the negative freq and prob
                combine[index].append(copy.deepcopy(item[1]))
                combine[index].append(copy.deepcopy(item[2]))
            else:
                newItem = [item[0], 0.0, new_word_prob_pos, item[1], item[2]]
                combine.append(newItem)
        # Looks for words that have no value for negative freq and prob
        for item in combine:
            if len(item) == 3:
                item.append("0.0")
                item.append(new_word_prob_neg)
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

    # Writes removed words to file
    def writeToRemoveFile(self, fileName):
        f = open(fileName + ".txt", "w", encoding='utf-8')
        for item in self.removed_words_list:
            f.write(str(item) + "\n")
        f.close()

    def removeWordsByFrequency(self, arg):
        if arg == '=1':
            # go through pos_info and neg_info, adding in removed_words_list with freq = 1 words
            self.addWordsLessThanFrequencyToRemoveList(1)
        if arg == '<=10':
            self.addWordsLessThanFrequencyToRemoveList(10)
        if arg == '<=20':
            self.addWordsLessThanFrequencyToRemoveList(20)
        if arg == 'top5%':
            self.addWordsTopPercentageToRemoveList(0.05)
        if arg == 'top10%':
            self.addWordsTopPercentageToRemoveList(0.1)
        if arg == 'top20%':
            self.addWordsTopPercentageToRemoveList(0.2)
        self.updateWordInfoWithRemovedWords()

    # Add all the words to removed_word_list if it's combined frequency is less than freq parameter
    def addWordsLessThanFrequencyToRemoveList(self, freq):
        for item in self.allWordInfo:
            if self.__getCombinedFrequencyOfWord(item) <= freq:
                self.removed_words_list.append(item[0])

    # Add all the words to removed_word_list if it's combined frequency is greater than freq parameter
    def addWordsGreaterThanFrequencyToRemoveList(self, freq):
        for item in self.allWordInfo:
            if self.__getCombinedFrequencyOfWord(item) >= freq:
                self.removed_words_list.append(item[0])

    # Add all the words to removed_word_list if it's combined frequency is greater than percentage param
    def addWordsTopPercentageToRemoveList(self, percentage):
        cut_off_freq = self.__getCutOffFrequencyWithTopPercentage(percentage)
        self.addWordsGreaterThanFrequencyToRemoveList(cut_off_freq)
        pass

    # Helper for addWordsTopPercentageToRemoveList()
    def __getCutOffFrequencyWithTopPercentage(self, percentage):
        # sort allWordInfo according to combined frequency of each word, descending order (most frequent in the front)
        self.allWordInfo.sort(key=lambda item: self.__getCombinedFrequencyOfWord(item), reverse=True)
        cut_off_index = int(percentage * len(self.allWordInfo))
        if cut_off_index <= len(self.allWordInfo)-1:  # Make sure no array out of bound
            return self.__getCombinedFrequencyOfWord(self.allWordInfo[cut_off_index])
        else:
            return float("inf")
        pass

    # Updating each word's cond probability and remove words with removed_words_list
    def updateWordInfoWithRemovedWords(self):
        # Remove from pos_info if this item is in the removed_word_set
        self.pos_info = [item for item in self.pos_info if item[0] not in self.removed_words_list]
        # need to update the conditional probability of each word
        self.updateProbabilityList(self.pos_info)
        self.neg_info = [item for item in self.neg_info if item[0] not in self.removed_words_list]
        self.updateProbabilityList(self.neg_info)
        # combine results
        self.allWordInfo = self.combinePosNegInfo()
        # new_word_set = self.getWordSet()

    def updateProbabilityList(self, info_list):
        count_list = [item[1] for item in info_list]
        prob_list = self.calculateProbabilityList(count_list)
        for i in range(len(info_list)):
            info_list[i][2] = prob_list[i]

    # Adding the positive and negative frequency together for an entry of allWordInfo
    @staticmethod
    def __getCombinedFrequencyOfWord(item):
        return float(item[1]) + float(item[3])

    # Return total vocabulary size
    def getVocabularySize(self):
        return len(self.allWordInfo)



