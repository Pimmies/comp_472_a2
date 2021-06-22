# comp_472_a2

Sources:
https://www.imdb.com/title/tt1475582/


### CONTRIBUTORS ###
Julie Pierrisnard - 40077165
Ningyuan Sun - 40124859


### BUILD INSTRUCTIONS ###
numpy, matplotlib, beautifulsoup4, requests, sklearn need to be installed.
https://numpy.org/
https://matplotlib.org/
https://pypi.org/project/beautifulsoup4/
https://docs.python-requests.org/en/master/
https://scikit-learn.org/stable/

The program is run by building python3 main.py in your IDE or
navigating to the file's directory in the command prompt then >>>python main.py


### PURPOSE ###
This program builds an Artificial Intelligence that determines whether an episode's review is positive or negative based on words used.
The AI uses Naive Bayes classifier model.
Here, we are predicting the review of episodes from the Sherlock series.


### FUNCTIONALITY ###
The program first scrapes the imdb website for reviews of all episodes of the sherlock series.
The reviews are divided between those that will be used to build the model and those that will be used for testing. (See generated folders "./testing_set" and "./training_set")

A model is built with all words in the training set with smoothing = 1. See model.py for the Model class.
A tester uses the model to predict the outcome of the reviews from the testing set. See tester.py for the Tester class.
The model and result outcomes are printed in "model.txt" and "result.txt".

More models are built by gradually removing words with different frequencies.
Testers predict the outcome each time.
All the removed words are found in "frequency_remove.txt".
The final model and result are printed in "frequency_model.txt" and "frequency_result.txt".

Models are built by gradually increasing the smoothing value.
Testers predict the outcome each time.
The model for smoothing = 1.6 and result for smoothing = 1.6 are printed in "smoothing_model.txt" and "smoothing_result.txt".

A graph is made that plots the prediction accurary using the F-measure method against the total vocabulary size.
Another graph plots the F-measure against the smoothing value.

Conclusion: In our case, bigger vocabulary size produces more accurate predictions. Increasing smoothing values do not produce notable changes on the prediction accuracy.
