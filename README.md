![music_stock_photo](https://github.com/slarionne/Spotify_Analysis_ML/assets/15343933/33668784-c4e7-403b-9c85-1159037198b6)

# Spotify_Analysis_ML
This repository contains Python code for analyzing the Spotify Songs dataset using machine learning models. The dataset, sourced from TidyTuesday, includes various features of Spotify songs, such as danceability, energy, key, mode, instrumentalness, liveness, valence, tempo, and duration.

## Table of Contents
### Data Loading and Exploration
Load the dataset from a linked file using pandas.
Display basic information about the dataset and explore unique values in selected columns.

### Data Preprocessing
Select relevant columns for analysis.
Normalize the selected variables using Min-Max scaling.
Visualize histograms of the original and scaled variables for comparison.

### Decision Tree Classification
Prepare the data for classification, considering the prediction of song modality (Minor or Major).
Split the data into training and testing sets.
Determine the optimal depth for the decision tree using a range of values.
Train the decision tree classifier and visualize the resulting tree.
Evaluate the classification metrics and accuracy.

### Random Forest Classification
Prepare the data for classification, focusing on predicting song popularity.
Simplify the classification by defining class boundaries and converting labels to numerical values.
Split the data into training and testing sets.
Build and train a random forest classifier.
Evaluate the classifier's accuracy.
Calculate and visualize feature importances.

### Conclusion
The code provides a comprehensive analysis of the Spotify Songs dataset, showcasing the application of decision tree and random forest classifiers for classification tasks. The results and visualizations offer insights into the importance of different features in predicting song modality and popularity.
