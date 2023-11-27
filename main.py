import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, train_test_split

df = pd.read_csv('https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-01-21/spotify_songs.csv')

columns = [
#     'track_id',
#     'track_name',
#     'track_artist',
#     'track_popularity',
#     'track_album_id',
#     'track_album_name',
#     'track_album_release_date',
#     'playlist_name',
#     'playlist_id',
#     'playlist_genre',
#     'playlist_subgenre',
    'danceability',
    'energy',
    'key',
#     'loudness',
    'mode',
#     'speechiness',
#     'acousticness',
    'instrumentalness',
    'liveness',
    'valence',
    'tempo',
    'duration_ms'
]

df_test = df[columns].head()

df.info()

# Major is represented by 1 and minor is 0.
df['mode'].value_counts()

for c in df[columns]:
    print(f'column: {c}, unique values: {df[c].nunique()}')

# Predictors
var = [
    'danceability',
    'energy',
    'key',
    'instrumentalness',
    'liveness',
    'valence',
    'tempo',
    'duration_ms'
]

varaibles_df = df[var]

# Normalize variables
scaler = MinMaxScaler()
scaled_df = scaler.fit_transform(df[var])


# histograms of the variables

df[var].hist()
plt.pyplot.show()

# converting to df to compare histograms with the original version

scaled_df2 = pd.DataFrame(scaled_df)

scaled_df2.hist()
plt.pyplot.show()




# We are trying to predict the modality of a song, Minor or Major
labels = df['mode']

# Data
data = scaled_df

# Split data into a train and test set
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, random_state=42, test_size=.2)

# Looking for the best depth
depths = range(1, 100)
acc_depth = []
for i in depths:
    dt = DecisionTreeClassifier(random_state = 42, max_depth = i)
    dt.fit(train_data, train_labels)
    acc_depth.append(dt.score(test_data, test_labels))


#Plot the accuracy vs depth
plt.plot(depths, acc_depth)
plt.xlabel('max_depth')
plt.ylabel('accuracy')
plt.show()

