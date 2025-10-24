import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import spacy
import nltk
from nltk.corpus import stopwords
import joblib

class ToxicityClassifier:
    def __init__(self):
        # Load spaCy Italian model
        self.nlp = spacy.load('it_core_news_sm')
        
        # Download Italian stopwords
        try:
            self.stop_words = set(stopwords.words('italian'))
        # If stopwords are not downloaded yet, download them
        except LookupError:
            nltk.download('stopwords')
            self.stop_words = set(stopwords.words('italian'))
        
        # Initialize vectorizer and classifier with parameters:
        # - max_features: limit to top 5000 features
        # - ngram_range: consider unigrams and bigrams
        # - min_df: ignore terms that appear in less than 2 documents
        # - max_df: ignore terms that appear in more than 95% of documents
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        self.classifier = LogisticRegression(random_state=42, max_iter=1000)
        # Paths for saving artifacts
        self.model_path = 'toxicity_logreg.joblib'
        self.vectorizer_path = 'tfidf_vectorizer.joblib'
    
    def clean_text(self, text):
        if pd.isna(text):
            return ""
        
        return text

    def preprocess_text(self, text):
        text = self.clean_text(text)
        
        if not text:
            return ""

        doc = self.nlp(text.lower())

        processed_tokens = []
        for token in doc:
            # Clean lemma
            lemma = token.lemma_.lower().strip()
            
            # Ignore punctuation, spaces, stop words, and short tokens
            if not token.is_punct and not token.is_space and lemma not in self.stop_words and len(lemma) > 2:
                processed_tokens.append(lemma)
                
        return ' '.join(processed_tokens)
    
    def load_data(self, csv_path):
        df = pd.read_csv(csv_path)

        texts = []
        labels = []
        
        for _, row in df.iterrows():
            if pd.notna(row['conversation']):
                texts.append(row['conversation'])
                labels.append(1)  # Toxic
            
            if pd.notna(row['Non-Toxic Conversation']):
                texts.append(row['Non-Toxic Conversation'])
                labels.append(0)  # Non-toxic

        print(f"Loaded Dataset: {len(texts)} samples")
        print(f"Toxic: {sum(labels)}, Non-toxic: {len(labels) - sum(labels)}")

        return texts, labels
    
    def preprocess_dataset(self, texts):
        processed_texts = []
        
        for i, text in enumerate(texts):
            if (i + 1) % 500 == 0:
                print(f"Preprocessing {i + 1}/{len(texts)}")
            processed_text = self.preprocess_text(text)
            processed_texts.append(processed_text)

        print("Preprocessing completed!")
        return processed_texts
    
    def train(self, texts, labels):
        # Preprocessing
        processed_texts = self.preprocess_dataset(texts)
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            processed_texts, labels, 
            test_size=0.2, 
            random_state=42, 
            stratify=labels
        )
        
        # Vectorizzazione
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)
        # Addestramento
        self.classifier.fit(X_train_tfidf, y_train)
        
        # Valutazione
        y_pred = self.classifier.predict(X_test_tfidf)
        
        print("\n=== Risultati ===")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(f"Precision: {precision_score(y_test, y_pred):.4f}")
        print(f"Recall: {recall_score(y_test, y_pred):.4f}")
        print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Non-Tossico', 'Tossico']))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        # Save for later use
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = y_pred
        self.X_test_tfidf = X_test_tfidf
        
        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred
        }
    
    
    def get_important_features(self, n_features=10):
        feature_names = self.vectorizer.get_feature_names_out()
        coef = self.classifier.coef_[0]
        
        # Most important features for toxic class (positive coefficients)
        toxic_indices = np.argsort(coef)[-n_features:][::-1]
        toxic_features = [(feature_names[i], coef[i]) for i in toxic_indices]
        
        # Most important features for non-toxic class (negative coefficients)
        nontoxic_indices = np.argsort(coef)[:n_features]
        nontoxic_features = [(feature_names[i], coef[i]) for i in nontoxic_indices]
        
        print("\n=== FEATURES PIÙ IMPORTANTI ===")
        print(f"\nTop {n_features} features per TOSSICO:")
        for feature, coef in toxic_features:
            print(f"  {feature}: {coef:.4f}")
        
        print(f"\nTop {n_features} features per NON-TOSSICO:")
        for feature, coef in nontoxic_features:
            print(f"  {feature}: {coef:.4f}")

    def save_model(self, model_path=None, vectorizer_path=None):
        model_path = model_path or self.model_path
        vectorizer_path = vectorizer_path or self.vectorizer_path
        joblib.dump(self.classifier, model_path)
        joblib.dump(self.vectorizer, vectorizer_path)
        print(f"Modello salvato su: {model_path}")
        print(f"Vectorizer salvato su: {vectorizer_path}")

if __name__ == "__main__":
    classifier = ToxicityClassifier()
    csv_path = 'datasets/classification_and_explaination_toxic_conversation_with_non_toxic.csv'
    texts, labels = classifier.load_data(csv_path)

    classifier.train(texts, labels)
    classifier.save_model()
    classifier.get_important_features()