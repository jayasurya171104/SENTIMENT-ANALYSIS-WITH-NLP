# SENTIMENT-ANALYSIS-WITH-NLP

COMPANY: CODTECH IT SOLUTIONS

NAME: JAYASURYA M

INTERN ID: CT04DF2950

DOMAIN: MACHINE LEARNING

DURATION: 4 WEEKS

MENTOR: NEELA SANTHOSH

## DESCRIOTON OF MY TASK

🧠 Project Title:
Sentiment Analysis on Customer Reviews using Natural Language Processing (NLP)

📌 Objective:
The goal of this project is to perform Sentiment Analysis on a dataset of customer reviews using TF-IDF Vectorization for feature extraction and Logistic Regression for classification. The system aims to classify the reviews into positive or negative sentiments.

🔧 Tools & Technologies Used:
Programming Language: Python

Libraries/Modules:

pandas, matplotlib, seaborn – Data handling & visualization

nltk – Natural language processing (stopword removal)

scikit-learn – TF-IDF Vectorization, Logistic Regression, Model evaluation

🔁 Workflow:
Data Loading:

The dataset (Test.csv) contains customer reviews with labeled sentiments.

Columns used: text (review), label (0 or 1)

Text Preprocessing:

Convert text to lowercase

Remove URLs, mentions, digits, and special characters

Remove stopwords using NLTK

Feature Extraction:

Use TfidfVectorizer to convert text into numerical feature vectors (sparse matrix)

Model Training:

Train a Logistic Regression model on the TF-IDF-transformed training data

Evaluation:

Evaluate the model using:

Accuracy Score

Classification Report (precision, recall, F1-score)

Confusion Matrix (visualized using seaborn heatmap)

📊 Confusion Matrix Visualization:
A confusion matrix heatmap is generated to show the distribution of correctly and incorrectly predicted labels.

📁 Deliverable:
A well-commented Jupyter Notebook that includes:

Preprocessing logic

Feature extraction

Model training & testing

Evaluation results with visualization

