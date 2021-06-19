import string

from Model import Model
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
    return reviews, count

# Training
basic_model = Model(training_pos_path, training_neg_path, 1)

# Start Testing
# if os.path.exists(testing_pos_path) and os.path.exists(testing_neg_path):  # - Check if testing data exists
#     pos_reviews, pos_reviews_count = readTestingReviewsFromFile(testing_pos_path)
#     print(pos_reviews)