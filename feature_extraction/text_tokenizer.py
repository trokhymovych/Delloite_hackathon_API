import string
import re

import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer

punctuations = string.punctuation

nlp = spacy.load('en')
stop_words = spacy.lang.en.stop_words.STOP_WORDS
parser = English()


def clear_text(text):
    text = re.sub(r"\n|\r|\t", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(" +", " ", text.strip())
    return text


def make_one_text(texts):
    final_text = ""

    for text in texts:
        final_text += clear_text(text) + " "

    return final_text.strip()


def transform(texts):
    return [clean_text(text) for text in texts]


def clean_text(text):
    return text.strip().lower()


def spacy_tokenizer(sentence):
    mytokens = parser(sentence)
    mytokens = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens]
    mytokens = [word for word in mytokens if word not in stop_words and word not in punctuations]
    return mytokens


bow_vector = CountVectorizer(tokenizer=spacy_tokenizer, ngram_range=(1, 1))
tfidf_vector = TfidfVectorizer(tokenizer=spacy_tokenizer)
hash_vector = HashingVectorizer(tokenizer=spacy_tokenizer, n_features=200)

