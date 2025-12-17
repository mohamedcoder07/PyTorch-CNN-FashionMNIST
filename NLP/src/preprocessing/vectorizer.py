from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator



class word_vectorizer(BaseEstimator):

    def __init__(self, method = "tfidf"):
        self.method = method
        self.vectorizer = None


        if self.method == "tfidf":
            self.vectorizer = TfidfVectorizer(stop_words = "english", min_df=5)
        
        elif self.method == "transformer" :
            self.vectorizer = None

        elif self.method == "elmo" :
            self.vectorizer = None


    def fit(self, texts):        
        self.vectorizer.fit(texts)
        return self
    

    def tranform(self, texts):
        
        if self.method == "tfidf":
            return self.vectorizer.transform(texts)
        
        elif self.method == "transformer":            
            return ""
        
    def fit_tranform(self, texts):
        
        if self.method == "tfidf":
            return self.vectorizer.fit_transform(texts)
        
        elif self.method == "transformer":            
            return ""        
        
    

