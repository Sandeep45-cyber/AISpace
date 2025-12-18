from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import numpy as np

class BaseClassifier:
    def train(self, texts, labels):
        raise NotImplementedError

    def predict(self, text):
        raise NotImplementedError

class KeywordClassifier(BaseClassifier):
    def __init__(self, keyword_map):
        """
        keyword_map: dict of {keyword: label}
        """
        self.keyword_map = keyword_map

    def train(self, texts, labels):
        # No training needed for rule-based
        pass

    def predict(self, text):
        text_lower = text.lower()
        for keyword, label in self.keyword_map.items():
            if keyword in text_lower:
                return label
        return "Unknown"

class NaiveBayesClassifier(BaseClassifier):
    def __init__(self):
        self.model = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', MultinomialNB()),
        ])

    def train(self, texts, labels):
        self.model.fit(texts, labels)

    def predict(self, text):
        return self.model.predict([text])[0]

class LogisticClassifier(BaseClassifier):
    def __init__(self):
        self.model = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', LogisticRegression(max_iter=1000)),
        ])

    def train(self, texts, labels):
        self.model.fit(texts, labels)

    def predict(self, text):
        return self.model.predict([text])[0]
