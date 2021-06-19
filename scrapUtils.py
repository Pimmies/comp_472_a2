from requests import get
from bs4 import BeautifulSoup
import re
import csv
import os


def createDataSets():
    #############################################################################################
    # 1.1 CSV
    # cvs containers
    csv_fields = ["Episode Name", "Season", "Review Link", "Year"]
    csv_rows = []
    
    season_urls = ['https://www.imdb.com/title/tt1475582/episodes?season=1',
                   'https://www.imdb.com/title/tt1475582/episodes?season=2',
                   'https://www.imdb.com/title/tt1475582/episodes?season=3',
                   'https://www.imdb.com/title/tt1475582/episodes?season=4']

    # Extract information from URLs
    season_counter = 0
    for url in season_urls:
        season_counter = season_counter + 1

        response = get(url)
        html_soup = BeautifulSoup(response.text, 'html.parser')  # creates beautifulsoup object from URL

        episode_container = html_soup.find_all('div', class_='list_item')  # gets all the episodes from that season

        for episode in episode_container:  # gets all the fields of that episode
            new_row = []
            name = episode.strong.a.text
            new_row.append(name)
            new_row.append(season_counter)
            link = "https://www.imdb.com" + episode.strong.a.get('href') + "reviews"
            new_row.append(link)
            date = episode.find('div', class_="airdate").text
            year = re.search(r"(\d{4})", date).group(1)  # extract just the year
            new_row.append(year)
            csv_rows.append(new_row)

    # Write to csv file
    csv_fileName = "data.csv"
    with open(csv_fileName, 'w') as csvfile:
        csv_writer = csv.writer(csvfile)  # creating a csv writer object
        csv_writer.writerow(csv_fields)
        csv_writer.writerows(csv_rows)

    ###########################################################################################
    # 1.1 BUILDING A TRAINING SET AND TESTING SET
    # makes:
    # a training set with positive and negative reviews
    # a testing set with positive and negative reviews
    # not yet tokenized

    # creates directories
    training_path = "./training_set"
    testing_path = "./testing_set"
    try:
        os.mkdir(training_path)
        os.mkdir(testing_path)
    except OSError:
        print("Creation of the directory failed. Directory might already exist.")
    else:
        print("Successfully created the directory")

    # removes existing files
    training_files = os.listdir(training_path)
    for item in training_files:
        if item.endswith(".txt"):
            os.remove(os.path.join(training_path, item))
    testing_files = os.listdir(testing_path)
    for item in testing_files:
        if item.endswith(".txt"):
            os.remove(os.path.join(testing_path, item))

    # Get reviews of each episode and write to files
    pos_training_flag = True
    neg_training_flag = True
    for episode in csv_rows:
        response = get(episode[
                           2] + '?spoiler=hide&sort=helpfulnessScore&dir=desc&ratingFilter=0')  # get the episode review link and filter out the spoilers
        html_soup = BeautifulSoup(response.text, 'html.parser')  # creates beautifulsoup object from URL
        review_container = html_soup.find_all('div',
                                              class_="review-container")  # contains all reviews on the first page of the episode
        for review in review_container:
            if review.span.span is not None:  # only taking reviews that have a number rating
                if int(review.span.span.text) < 8:  # negative review
                    if neg_training_flag:  # alternates between training set and testing set
                        f = open("./training_set/training_negative.txt", "a", encoding='utf-8')
                    else:
                        f = open("./testing_set/testing_negative.txt", "a", encoding='utf-8')
                        f.write(review.a.text)  # review title
                    f.write(review.find('div', class_="text show-more__control").text + "\n")  # review text
                    f.write('/\n')
                    f.close()
                    neg_training_flag = not neg_training_flag
                else:  # positive review
                    if pos_training_flag:  # alternates between training set and testing set
                        f = open("./training_set/training_positive.txt", "a", encoding='utf-8')
                    else:
                        f = open("./testing_set/testing_positive.txt", "a", encoding='utf-8')
                        f.write(review.a.text)  # review title
                    f.write(review.find('div', class_="text show-more__control").text + "\n")
                    f.write('/\n')
                    f.close()
                    pos_training_flag = not pos_training_flag