import pandas as pd
import re
import nltk
import pickle
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# 1. Load Data (Ensure these are downloaded locally)
print("Loading datasets...")
true_news = pd.read_csv('True.csv')
fake_news = pd.read_csv('Fake.csv')

true_news['label'] = 1
fake_news['label'] = 0
news_pd = pd.concat([true_news, fake_news], axis=0).reset_index(drop=True)

# 2. Preprocess Text
print("Cleaning text...")
news_pd['title'] = news_pd['title'].fillna('').astype(str)
news_pd['text'] = news_pd['text'].fillna('').astype(str)
news_pd['content'] = news_pd['title'] + " " + news_pd['text']

nltk.download('stopwords', quiet=True)
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def stemming(content):
    content = re.sub('[^a-zA-Z]', " ", content)
    content = content.lower().split()
    content = [ps.stem(word) for word in content if word not in stop_words]
    return " ".join(content)

news_pd['content'] = news_pd['content'].apply(stemming)

# 3. Vectorize and Train
print("Vectorizing and training model...")
X = news_pd['content'].values
Y = news_pd['label'].values

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=1)

model = LogisticRegression()
model.fit(X_train, Y_train)

# 4. Export the artifacts
print("Exporting model and vectorizer...")
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("✅ Build complete! model.pkl and vectorizer.pkl generated.")
