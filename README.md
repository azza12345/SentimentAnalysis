# SentimentAnalysis
# Movie Review Sentiment Analysis

This repository contains code for sentiment analysis on movie reviews using various machine learning models.

## Dataset

The dataset used is the Movie Reviews dataset from the NLTK (Natural Language Toolkit) library. It consists of 2000 movie reviews labeled as positive or negative.

### Dataset Details:
- **Source**: NLTK movie_reviews corpus
- **Size**: 2000 movie reviews (sampled from the full dataset)
- **Classes**: Positive (1) and Negative (0)
- **Format**: Text reviews with sentiment labels

The dataset is pre-processed and saved as a CSV file in the `/data` directory.

## Models

This project implements several machine learning models for sentiment classification:

1. **Multinomial Naive Bayes**: A probabilistic classifier based on Bayes' theorem suitable for text classification.
2. **Logistic Regression**: A linear model that works well for binary classification tasks.
3. **Support Vector Machine (LinearSVC)**: A powerful classifier that finds the optimal hyperplane to separate classes.
4. **Random Forest**: An ensemble learning method based on decision trees.

## Feature Extraction

Text data is converted to numerical features using:
- **TF-IDF Vectorization**: Transforms text into numerical features based on term frequency and inverse document frequency.
- Stop words removal and feature selection are applied to improve model performance.

## Evaluation

Models are evaluated using:
- Accuracy score
- Classification report (precision, recall, F1-score)
- Confusion matrix

## Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn
- nltk

## Usage

1. Run `extract_dataset.py` to download and prepare the dataset
2. Run `sentiment_analysis.py` to train and evaluate the models
3. Modify the model selection in the code to experiment with different classifiers

## License

This project is available under the MIT License.
