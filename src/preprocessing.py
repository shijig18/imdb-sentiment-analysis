import re
import emoji
import contractions
import spacy
from bs4 import BeautifulSoup

nlp = spacy.load("en_core_web_sm")

stop_words = set([
    'the','is','in','and','a','to','of','it','that','this','was','for','on','with',
    'as','but','are','very','so','because'
])

def remove_html(text):
    return BeautifulSoup(text, "html.parser").get_text()

def preprocess_text(text):
    text = text.lower()
    text = remove_html(text)
    text = contractions.fix(text)
    text = emoji.replace_emoji(text, replace='')
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    doc = nlp(text)

    tokens = [
        token.lemma_
        for token in doc
        if token.text not in stop_words and token.is_alpha
    ]

    return " ".join(tokens)