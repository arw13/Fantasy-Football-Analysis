"Import nfl game data into csv and then convert to pandas dataframe"

import nflgame

# For each season, save to a separate csv file
for i in range(2009,2018):
    filename = "season"+str(i)+".csv"
    nflgame.combine(nflgame.games(i)).csv(filename)
