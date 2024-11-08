import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

class IRSystem:
    def __init__(self, n_samples=100):
        self.n_samples = n_samples
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.documents = None
        self.document_vectors = None
        
    def preprocess_text(self, text):
        """Clean and preprocess text data"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def load_data(self, documents, summaries):
        """Load and preprocess the first n_samples documents and summaries"""
        self.documents = [self.preprocess_text(doc) for doc in documents[:self.n_samples]]
        self.summaries = [self.preprocess_text(summary) for summary in summaries[:self.n_samples]]
        
        # Create TF-IDF vectors for documents
        self.document_vectors = self.vectorizer.fit_transform(self.documents)
        
    def retrieve(self, query, top_k=None):
        """
        Retrieve and rank documents based on a query
        Returns: List of (document_index, similarity_score) tuples
        """
        # Preprocess query
        query = self.preprocess_text(query)
        
        # Transform query using the same vectorizer
        query_vector = self.vectorizer.transform([query])
        
        # Calculate cosine similarity between query and all documents
        similarities = cosine_similarity(query_vector, self.document_vectors).flatten()
        
        # Sort documents by similarity score
        ranked_docs = [(idx, score) for idx, score in enumerate(similarities)]
        ranked_docs.sort(key=lambda x: x[1], reverse=True)
        
        if top_k:
            ranked_docs = ranked_docs[:top_k]
            
        return ranked_docs
    
    def evaluate(self):
        """
        Evaluate the IR system using the summaries as queries
        Returns: Mean Reciprocal Rank (MRR) score
        """
        reciprocal_ranks = []
        
        for idx, summary in enumerate(self.summaries):
            # Use each summary as a query
            ranked_docs = self.retrieve(summary)
            
            # Find the position of the correct document
            for rank, (doc_idx, _) in enumerate(ranked_docs, 1):
                if doc_idx == idx:
                    reciprocal_ranks.append(1.0 / rank)
                    break
                    
        mrr = np.mean(reciprocal_ranks)
        return mrr

