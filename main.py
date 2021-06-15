from requests import get
from bs4 import BeautifulSoup
import re
import csv

#containers
csv_fields = ["Episode Name", "Season", "Review Link", "Year"]
csv_rows = []
season_urls = ['https://www.imdb.com/title/tt1475582/episodes?season=1', 'https://www.imdb.com/title/tt1475582/episodes?season=2', 'https://www.imdb.com/title/tt1475582/episodes?season=3', 'https://www.imdb.com/title/tt1475582/episodes?season=4']

#extract information to write to the csv file
season_counter = 0
for url in season_urls:
	season_counter = season_counter +1
	
	response = get(url)
	html_soup = BeautifulSoup(response.text, 'html.parser') #creates beautifulsoup object from URL

	episode_container = html_soup.find_all('div', class_ = 'list_item') #gets all the episodes from that season

	for episode in episode_container: #gets all the fields of that episode
		new_row = []
		name = episode.strong.a.text
		new_row.append(name)
		new_row.append(season_counter)
		link = "https://www.imdb.com" + episode.strong.a.get('href') + "reviews"
		new_row.append(link)
		date = episode.find('div', class_ = "airdate").text
		year = re.search(r"(\d{4})", date).group(1) #extract just the year
		new_row.append(year)
		csv_rows.append(new_row)

# writing to csv file 
csv_fileName = "data.csv"
with open(csv_fileName, 'w') as csvfile: 
    csv_writer = csv.writer(csvfile) # creating a csv writer object 
    csv_writer.writerow(csv_fields)  
    csv_writer.writerows(csv_rows)
