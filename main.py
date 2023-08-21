import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# Load the dataset from the CSV file
df = pd.read_csv('news.csv')

# Print the shape and first few rows of the dataset
print("Dataset Shape:", df.shape)
print("First Few Rows:")
print(df.head())

# Extract labels from the dataset
labels = df.label

# Print the first few labels
print("First Few Labels:")
print(labels.head())

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(df['text'], labels, test_size=0.2, random_state=7)

# Initialize a TF-IDF vectorizer with additional preprocessing options
tfidf_vectorizer = TfidfVectorizer(
    stop_words='english',         # Remove common English stop words
    max_df=0.7,                   # Ignore terms that appear in more than 70% of documents
    ngram_range=(1, 2),           # Consider both unigrams and bigrams
    max_features=5000             # Limit the number of features to 5000
)

# Transform the training and testing text data into TF-IDF representations
tfidf_train = tfidf_vectorizer.fit_transform(x_train)
tfidf_test = tfidf_vectorizer.transform(x_test)

# Initialize a Passive Aggressive Classifier with more parameters
pac = PassiveAggressiveClassifier(
    C=1.0,                        # Regularization parameter
    fit_intercept=True,           # Include an intercept term in the decision function
    shuffle=True,                 # Shuffle the training data before each epoch
    n_iter_no_change=10,          # Number of iterations with no improvement to wait before stopping
    validation_fraction=0.1,      # Fraction of training data to set aside as validation set
    n_jobs=-1                     # Use all available CPU cores for training
)

# Train the Passive Aggressive Classifier on the TF-IDF training data
pac.fit(tfidf_train, y_train)

# Predict labels on the TF-IDF testing data
y_pred = pac.predict(tfidf_test)

# Calculate and print the accuracy of the classifier
score = accuracy_score(y_test, y_pred)
print(f'Accuracy: {round(score * 100, 2)}%')

# Calculate and print the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL'])
print("Confusion Matrix:")
print(conf_matrix)

# Visualize the distribution of predicted labels and original labels using a bar plot
plt.figure(figsize=(8, 6))
labels = ['FAKE', 'REAL']
pred_counts = [np.sum(y_pred == label) for label in labels]
true_counts = [np.sum(y_test == label) for label in labels]

plt.bar(labels, pred_counts, color='blue', alpha=0.6, label='Predicted')
plt.bar(labels, true_counts, color='green', alpha=0.6, label='True')

plt.xlabel('Labels')
plt.ylabel('Count')
plt.title('Distribution of Predicted and True Labels')
plt.legend()

plt.show()
