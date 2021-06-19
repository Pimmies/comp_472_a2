from sklearn.feature_extraction.text import CountVectorizer
import string
import re
import math

# Converting a txt file of testing reviews in to a list of strings in memory
def readTestingReviewsFromFile(path):
    file = open(path, 'r')
    reviews = []
    count = 0
    isTitleFlag = True
    title = ""
    content = ""
    while True:
        # Get next line from file
        line = file.readline()
        if line == '/\n':  # If the line is a /, it means a review is finished
            count = count + 1
            reviews.append((title, content))
            isTitleFlag = True  # The next line will be a review title
            content = ""
        else:
            if isTitleFlag:
                title = line
                isTitleFlag = False
            else:
                content = content + " " + line
        if not line:  # if line is empty end of file is reached
            break
    file.close()
    return reviews #, count

# Tokenizing a string, first strip all the punctuation and then return a list of words
def tokenize(txt):
    return txt.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation))).split()

# Finds a whole word rather than a substring
def findWholeWord(w):
    return re.compile(r'\b({0})\b'.format(w), flags=re.IGNORECASE).search

class Tester:
    """This class tests reviews"""
    def __init__(self, testing_pos_path, testing_neg_path, model):
        # list of reviews, and review count
        self.pos_reviews = readTestingReviewsFromFile(testing_pos_path)
        self.neg_reviews = readTestingReviewsFromFile(testing_neg_path)
        self.model = model
        self.results = self.runTest()
        self.writeTestResultsToFile("result")

    # Returns list of tokenized words
    def getWordsInReview(self, review):
        # Custom tokenizer overrides the default which ignore 1 letter word
        vec = CountVectorizer(tokenizer=lambda txt: tokenize(txt))
        l = []
        l.append(review)
        X = vec.fit_transform(l)
        word_list = vec.get_feature_names()
        return word_list

    # Builds a result list [[review title, positive score, negative score, our result, correct result, right or wrong], [...]]
    def runTest(self):
        result_list = []
        success = 0
        failure = 0
        for review in self.pos_reviews:
            word_list = self.getWordsInReview(review[1]) # build word list based on review text (not title)
            score_pos, score_neg = self.predictScores(word_list)
            if(score_pos > score_neg):
                result_list.append([review[0], score_pos, score_neg, "Positive", "Positive", "Right"])
                success += 1
            else:
                result_list.append([review[0], score_pos, score_neg, "Negative", "Positive", "Wrong"])
                failure += 1
        # Predict negative reviews
        for review in self.neg_reviews:
            word_list = self.getWordsInReview(review[1]) # build word list based on review text (not title)
            score_pos, score_neg = self.predictScores(word_list)
            if(score_neg > score_pos):
                result_list.append([review[0], score_pos, score_neg, "Negative", "Negative", "Right"])
                success += 1
            else:
                result_list.append([review[0], score_pos, score_neg, "Positive", "Negative", "Wrong"])
                failure += 1
        print("number of successes: ", success)
        print("number of failures: ", failure)
        return result_list

    # Returns positive and negative scores for a review's list of words
    def predictScores(self, word_list):
        positive_score = math.log10(self.model.pos_reviews_count / (self.model.pos_reviews_count + self.model.neg_reviews_count))
        negative_score = math.log10(self.model.neg_reviews_count / (self.model.pos_reviews_count + self.model.neg_reviews_count))
        new_word_prob_pos = self.model.newWordProbability("positive")
        new_word_prob_neg = self.model.newWordProbability("negative")
        for word in word_list:
            index = None
            for item in self.model.allWordInfo: # Look first if the word is already in the model
                if findWholeWord(word)(item[0]):
                    index = self.model.allWordInfo.index(item)
            if index is not None:
                positive_score += self.model.allWordInfo[index][2]
                negative_score += self.model.allWordInfo[index][4]
            else: # new word
                positive_score += new_word_prob_pos
                negative_score += new_word_prob_neg
        return (positive_score, negative_score)

    def writeTestResultsToFile(self, fileName):
        f = open(fileName + ".txt", "w", encoding='utf-8')
        counter = 1
        for item in self.results:
            f.write("No." + str(counter) + item[0])
            f.write(str(item[1]) + ', ' + str(item[2]) + ', ' + item[3] + ', ' + item[4] + ', ' + item[5] + '\n\n')
            counter += 1
        f.close()