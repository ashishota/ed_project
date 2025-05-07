import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load data
df = pd.read_csv("career_recommender.csv")
df.fillna('', inplace=True)

# Select relevant columns
df.columns = df.columns.str.strip()
df = df[['What are your interests?',
         'What are your skills ? (Select multiple if necessary)',
         'If yes, then what is/was your first Job title in your current field of work? If not applicable, write NA.']]

# Rename for simplicity
df.columns = ['Interests', 'Skills', 'JobTitle']

# Text preprocessing (without stemming)
stop_words = set(stopwords.words('english'))

def clean(text):
    # Remove non-alphanumeric characters except spaces (preserve spaces)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    
    # Tokenize the cleaned text and convert to lowercase
    tokens = nltk.word_tokenize(text.lower())
    
    # Remove stopwords
    return " ".join([word for word in tokens if word not in stop_words])

# Apply the cleaning function to the relevant columns
for col in ['Interests', 'Skills', 'JobTitle']:
    df[col] = df[col].astype(str).apply(clean)

# Combine all into one field
df['clean_text'] = df['Interests'] + " " + df['Skills'] + " " + df['JobTitle']

# Train TF-IDF and compute similarity matrix
tfidf = TfidfVectorizer(stop_words='english')
matrix = tfidf.fit_transform(df['clean_text'])
similarity = cosine_similarity(matrix)

# Save artifacts for later use in app.py
pickle.dump(df, open('df.pkl', 'wb'))
pickle.dump(tfidf, open('vectorizer.pkl', 'wb'))
pickle.dump(similarity, open('similarity.pkl', 'wb'))
pickle.dump(matrix, open('matrix.pkl', 'wb'))
