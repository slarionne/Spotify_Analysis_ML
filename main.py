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

# Find the largest accuracy and the depth at with it occurs
max_acc = np.max(acc_depth)
best_depth = depths[np.argmax(acc_depth)]
print(f'Highest accuracy {round(max_acc,4)}% at depth {best_depth}')

# Decision Tree Alg Fit
dt_final = DecisionTreeClassifier(random_state = 42,
                                  max_depth = best_depth,
                                  max_leaf_nodes=17)
dt_final.fit(train_data, train_labels)

# Predict
y_pred = dt_final.predict(test_data)

metrics = classification_report(test_labels, y_pred)
print("The metrics for a classification model")
print(metrics)

plt.figure(figsize=(16,10))
tree.plot_tree(dt_final, feature_names = var,
               class_names = ['Minor', 'Major'], # Major is represented by 1 and minor is 0.
                filled=True)
plt.show()

# Predicting popularity

#Random Forest Classification

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Splitting features and target
x_data = df.drop('track_popularity',axis = 1)
y_data = df['track_popularity']

print(f'Shape of x_train {x_data.shape} and y_data {y_data.shape}')


# Removing all identifying features

"""All identifying features have the dtype of Object, and the only potentially useful features with that dtype are 
Genre and Subgenre. However, for now, all "Objects" are excluded."""


x_data = x_data.select_dtypes(exclude=['object'])
x_data.head()

sns.set(rc={'figure.figsize': (20, 16)})
x_data.hist(color='skyblue')

y_data.hist(color='orange',bins = 50)

# Simplify Classification

# Define class boundaries
class_boundaries = [y_data.median(),100]

# Create a new column in y_data representing the simplified classes
y_data_class = np.digitize(y_data, bins=class_boundaries, right=True)

# Convert class labels to numerical values
num_classes = len(np.unique(y_data_class))

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data_class, test_size=0.2, random_state=42)


# Building Model

# Build a random forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier
rf_classifier.fit(x_train, y_train)

# Make predictions on the test set
y_pred = rf_classifier.predict(x_test)

# Evaluate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Random forest Accuracy:", accuracy)

# Calculating feature importance

feature_importances = rf_classifier.feature_importances_

# Sorting the feature importances in descending order
sorted_idx = feature_importances.argsort()

# Plotting
plt.figure(figsize=(10, 8))
plt.barh(range(len(sorted_idx)), feature_importances[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), [x_data.columns[i] for i in sorted_idx])
plt.title('Feature Importances in Random Forest Model')
plt.xlabel('Relative Importance')
plt.ylabel('Feature')
plt.show()

"""
Results shows that the most import features include, loudness, diration_ms, tempo, and energy.

"""

