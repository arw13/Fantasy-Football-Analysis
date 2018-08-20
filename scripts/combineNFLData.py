""" Script to combine the collected data into a single training set """

import pandas as pd 

# load in data to use in training
data_dir = '../sortedSeasonData/'
df2009 = pd.read_csv(data_dir+'sortedSeason2009.csv')
df2010 = pd.read_csv(data_dir+'sortedSeason2009.csv')
df2011 = pd.read_csv(data_dir+'sortedSeason2009.csv')
df2012 = pd.read_csv(data_dir+'sortedSeason2009.csv')
df2013 = pd.read_csv(data_dir+'sortedSeason2009.csv')
df2014 = pd.read_csv(data_dir+'sortedSeason2009.csv')
df2015 = pd.read_csv(data_dir+'sortedSeason2009.csv')
df2016 = pd.read_csv(data_dir+'sortedSeason2009.csv')
df2017 = pd.read_csv(data_dir+'sortedSeason2009.csv')


#2009
# create tensors to hold input and outputs
plyr2009 = df2009.iloc[:,:-2]
pts2009 = pd.concat([df2010.iloc[:,1], df2010.iloc[:,-1]], axis=1)
# join so that the training data matches the outputs
df2009 = pd.concat([plyr2009, pts2009], axis=1, join='inner')

#2010
# create tensors to hold input and outputs
plyr2010 = df2010.iloc[:,:-2]
pts2010= pd.concat([df2011.iloc[:,1], df2011.iloc[:,-1]], axis=1)
# join so that the training data matches the outputs
df2010 = pd.concat([plyr2010, pts2010], axis=1, join='inner')

#2011
# create tensors to hold input and outputs
plyr2011 = df2011.iloc[:,:-2]
pts2011 = pd.concat([df2012.iloc[:,1], df2012.iloc[:,-1]], axis=1)
# join so that the training data matches the outputs
df2011 = pd.concat([plyr2011, pts2011], axis=1, join='inner')

#2012
# create tensors to hold input and outputs
plyr2012 = df2012.iloc[:,:-2]
pts2012 = pd.concat([df2013.iloc[:,1], df2013.iloc[:,-1]], axis=1)
# join so that the training data matches the outputs
df2012 = pd.concat([plyr2012, pts2012], axis=1, join='inner')

#2013
# create tensors to hold input and outputs
plyr2013 = df2013.iloc[:,:-2]
pts2013 = pd.concat([df2014.iloc[:,1], df2014.iloc[:,-1]], axis=1)
# join so that the training data matches the outputs
df2013 = pd.concat([plyr2013, pts2013], axis=1, join='inner')

#2014
# create tensors to hold input and outputs
plyr2014 = df2014.iloc[:,:-2]
pts2014 = pd.concat([df2015.iloc[:,1], df2015.iloc[:,-1]], axis=1)
# join so that the training data matches the outputs
df2014 = pd.concat([plyr2014, pts2014], axis=1, join='inner')

#2015
# cre1te tensors to hold input and outputs
plyr2015 = df2015.iloc[:,:-2]
pts2015 = pd.concat([df2016.iloc[:,1], df2016.iloc[:,-1]], axis=1)
# join so that the training data matches the outputs
df2015 = pd.concat([plyr2015, pts2015], axis=1, join='inner')

#2016
# create tensors to hold input and outputs
plyr2016 = df2016.iloc[:,:-2]
pts2016 = pd.concat([df2017.iloc[:,1], df2017.iloc[:,-1]], axis=1)
# join so that the training data matches the outputs
df2016 = pd.concat([plyr2016, pts2016], axis=1, join='inner')

df_data = pd.concat([df2009, df2010, df2011, df2012, df2013, df2014, df2015, df2016])

# Tried to remove id duplicate column, but removed both. replace if needed
df_data = df_data.drop(df_data.columns[-2], axis =1)
df_data.to_csv('../sortedSeasonData/allSeasons.csv', index=False)