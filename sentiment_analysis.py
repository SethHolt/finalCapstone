# Implement sentiment analysis model.
import spacy
import pandas as pd
from textblob import TextBlob

nlp = spacy.load("en_core_web_sm")

# Open reviews data.
df = pd.read_csv("Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv")

# Clean the data.
clean_data = df.dropna(subset=["reviews.text"])

def remove_stop_words(review):
    text = nlp(review)
    removed = [token for token in text if not token.is_stop]
    return " ".join([token.text for token in removed])

# Function to analyze sentiment using spaCytextblob. Also calls the function to remove stop words.
def analyse_sentiment(review):
    text = remove_stop_words(review)
    text = TextBlob(review)
    sentiment = text.sentiment.polarity
    if sentiment > 0:
        return "Positive"
    elif sentiment < 0:
        return "Negative"
    else:
        return "Neutral"

# Iterate through each review.
for index,row in clean_data.iterrows():
    review = row["reviews.text"]
    sentiment = analyse_sentiment(review)

    # Print each review with it's sentiment.
    print(f"Review {index + 1}: \n {review} \n Sentiment: {sentiment} \n")

# Analyse the similarity of two random reviews.
choice_1 = nlp(df["reviews.text"][2000])
choice_2 = nlp(df["reviews.text"][10000])

print(choice_1.similarity(choice_2))