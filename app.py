import streamlit as st
import pickle
import re
import requests
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Setup page config
st.set_page_config(page_title="Live News AI Validator", page_icon="🕵️", layout="centered")

# Download stopwords securely for cloud deployment
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

# Load the ML artifacts
@st.cache_resource
def load_assets():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

model, vectorizer = load_assets()

# Preprocessing function for incoming live text
ps = PorterStemmer()
def clean_text(content):
    content = re.sub('[^a-zA-Z]', " ", content)
    content = content.lower().split()
    content = [ps.stem(word) for word in content if word not in stopwords.words('english')]
    return " ".join(content)

# API Fetch function
def fetch_live_news(query, api_key):
    url = f"https://newsapi.org/v2/everything?q={query}&apiKey={api_key}&language=en&sortBy=publishedAt&pageSize=1"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if data.get('articles'):
                article = data['articles'][0]
                return article['title'], article['description'], article['url']
    except Exception as e:
        st.error(f"API Error: {e}")
    return None, None, None

# UI Construction
st.title("🕵️ Live AI News Validator")
st.markdown("Enter a topic below to fetch the latest breaking news. The underlying NLP model will evaluate the content's linguistic patterns to estimate credibility.")

topic = st.text_input("Enter a news topic (e.g., 'Economy', 'Technology', 'Elections'):")

# Retrieve the API key from Streamlit's secure secrets management
API_KEY = st.secrets["0a2f5cd9d0b14a15ab667083fd6e0506"] if "NEWS_API_KEY" in st.secrets else "YOUR_LOCAL_API_KEY"

if st.button("Fetch & Analyze Live News", type="primary"):
    if topic:
        with st.spinner(f"Scraping latest news for '{topic}'..."):
            title, description, url = fetch_live_news(topic, API_KEY)
            
            if title and description:
                # Display the fetched news
                st.subheader("📰 Latest Headline")
                st.write(f"**{title}**")
                st.write(f"> *{description}*")
                st.caption(f"[Read full source article]({url})")
                
                # NLP Processing
                full_text = str(title) + " " + str(description)
                processed_text = clean_text(full_text)
                vectorized_input = vectorizer.transform([processed_text])
                prediction = model.predict(vectorized_input)[0]
                
                # Verdict
                st.divider()
                if prediction == 1:
                    st.success("✅ **AI Verdict: REAL** - The linguistic patterns match factual reporting.")
                else:
                    st.error("🚨 **AI Verdict: FAKE** - High probability of sensationalism, bias, or misinformation.")
            else:
                st.warning("Could not find recent news for this topic. Try broader keywords.")
    else:
        st.info("Please enter a topic to begin.")
