""" nflDataOrg function script
    input: season .csv, name of output
    output: save season as an org
"""

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

def main():
    saved_data = [
        'season2009.csv',
        'season2010.csv',
        'season2011.csv',
        'season2012.csv',
        'season2013.csv',
        'season2014.csv',
        'season2015.csv',
        'season2016.csv',
        'season2017.csv',
        ]
    outputs = [
        'sortedSeason2009.csv',
        'sortedSeason2010.csv',
        'sortedSeason2011.csv',
        'sortedSeason2012.csv',
        'sortedSeason2013.csv',
        'sortedSeason2014.csv',
        'sortedSeason2015.csv',
        'sortedSeason2016.csv',
        'sortedSeason2017.csv',
    ]

    year = range(2009,2018)

    for i in range(len(saved_data)):
        dataOrg(saved_data[i], outputs[i], year[i])
    
    



def dataOrg(datafile, outfile, year):
    """ organizes data from nflgames to be used in ML analysis"""

    data_dir = '../seasonData/'
    output_dir = '../sortedSeasonData/'

    # read date
    df = pd.read_csv(data_dir+datafile)

    # remove defensive players
    df = df[df.defense_ast.isnull()]

    # remove punters
    df = df[df.punting_avg.isnull()]

    # remove kickers(not sure if kickers should be included here)
    df = df[df.kicking_xpb.isnull()]

    #  removing unused columns
    df = df.drop(labels=['home', 'pos','defense_ast', 'defense_ffum', 'defense_int', 'defense_sk', 'defense_tkl', 'punting_avg', 'punting_i20', 'punting_lng', 'punting_pts', 'punting_yds', 'puntret_avg', 'puntret_lng', 'puntret_ret', 'kickret_avg', 'kickret_lng', 'kickret_ret', 'rushing_twopta', 'receiving_twopta','kicking_fga','kicking_fgm', 'kicking_fgyds', 'kicking_totpfg', 'kicking_xpa', 'kicking_xpb','kicking_xpmade', 'kicking_xpmissed', 'kicking_xptot'], axis =1)

    # one hot encoding for players team

    df = pd.get_dummies(df, columns=["team"])

    # Zero NaNs
    df =  df.fillna(0)

    # Calculate fantasy points
    # passing yards
    points = df['passing_yds'].apply(lambda x: x/25)
    # passing TDs
    points = points + df['passing_tds'].apply(lambda x: x*4)
    # interceptions
    points = points + df['passing_ints'].apply(lambda x: x*-1)
    # rushing yards
    points = points + df['rushing_yds'].apply(lambda x: x/10)
    # rushing TDs
    points = points + df['rushing_tds'].apply(lambda x: x*6)
    # receiving yards
    points = points + df['receiving_yds'].apply(lambda x: x/10)
    # receiving tds
    points = points + df['receiving_tds'].apply(lambda x: x*6)
    # return tds
    points = points + df['kickret_tds'].apply(lambda x: x*10) + df['puntret_tds'].apply(lambda x: x*10)
    # 2 pt convs
    points = points + df['receiving_twoptm'].apply(lambda x: x*2) + df['rushing_twoptm'].apply(lambda x: x*2)
    # Fumbles lost
    points = points + df['fumbles_lost'].apply(lambda x: x*-2)

    "Add fantasy points column"
    df['year']    = year
    df['FtsyPts'] = points

    df.to_csv(output_dir + outfile, index=False)
    

if __name__ == '__main__':
    main()