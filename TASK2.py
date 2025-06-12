# Sentiment Analysis using TF-IDF Vectorization and Logistic Regression

# Step 1: Import Libraries
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# Step 2: Load Dataset (Twitter Sentiment Dataset)
df = pd.read_csv("/content/Test.csv")
df = df[['text', 'label']]
df.rename(columns={'text': 'text', 'label': 'sentiment'}, inplace=True)

# Step 3: Text Preprocessing
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)     # Remove URLs
    text = re.sub(r"@\w+", "", text)        # Remove mentions
    text = re.sub(r"[^a-z\s]", "", text)    # Remove special characters and digits
    text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])
    return text

df['text'] = df['text'].apply(clean_text)

# Step 4: Train-Test Split
X = df['text']
y = df['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Step 6: Train Logistic Regression Model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Step 7: Evaluate the Model
y_pred = model.predict(X_test_tfidf)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📄 Classification Report:\n", classification_report(y_test, y_pred))
print("\n📊 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Step 8: Confusion Matrix Visualization
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
