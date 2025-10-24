import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import spacy
import nltk
from nltk.corpus import stopwords
import joblib


class ToxicityClassifierKFold:
    def __init__(self, k_folds=5, random_state=42):
        # Load spaCy Italian model
        self.nlp = spacy.load('it_core_news_sm')
        
        # Download Italian stopwords
        try:
            self.stop_words = set(stopwords.words('italian'))
        except LookupError:
            nltk.download('stopwords')
            self.stop_words = set(stopwords.words('italian'))
        
        # Parameters
        self.k_folds = k_folds
        self.random_state = random_state
        
        # Initialize vectorizer and classifier
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        self.classifier = LogisticRegression(random_state=random_state, max_iter=1000)
        
        # K-Fold cross validation
        self.cv = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=random_state)
        
        # Paths for saving artifacts
        self.model_path = 'toxicity_logreg_kfold.joblib'
        self.vectorizer_path = 'tfidf_vectorizer_kfold.joblib'
        
        # Results storage
        self.cv_results = None
        self.final_results = None
    
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
            # Clean lemma from spaces and lower case
            lemma = token.lemma_.lower().strip()
            
            # Ignore punctuation, spaces, stop words and tokens that are too short
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
        print(f"Toxic: {sum(labels)}, Non-Toxic: {len(labels) - sum(labels)}")
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
    
    def cross_validate_detailed(self, texts, labels):
        print(f"Cross-validation with {self.k_folds} folds")
        
        # Preprocessing
        processed_texts = self.preprocess_dataset(texts)
        
        # Results storage
        fold_results = []
        all_predictions = []
        all_true_labels = []
        
        for fold, (train_idx, test_idx) in enumerate(self.cv.split(processed_texts, labels)):
            print(f"\n--- Fold {fold + 1}/{self.k_folds} ---")
            
            # Split data for this fold
            X_train_fold = [processed_texts[i] for i in train_idx]
            X_test_fold = [processed_texts[i] for i in test_idx]
            y_train_fold = [labels[i] for i in train_idx]
            y_test_fold = [labels[i] for i in test_idx]
            
            # Create fresh vectorizer for this fold
            vectorizer_fold = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95
            )
            classifier_fold = LogisticRegression(random_state=self.random_state, max_iter=1000)
            
            # Train on this fold
            X_train_tfidf = vectorizer_fold.fit_transform(X_train_fold)
            X_test_tfidf = vectorizer_fold.transform(X_test_fold)
            
            classifier_fold.fit(X_train_tfidf, y_train_fold)
            
            # Predict
            y_pred_fold = classifier_fold.predict(X_test_tfidf)
            
            # Calculate metrics for this fold
            accuracy = accuracy_score(y_test_fold, y_pred_fold)
            f1 = f1_score(y_test_fold, y_pred_fold)
            precision = precision_score(y_test_fold, y_pred_fold)
            recall = recall_score(y_test_fold, y_pred_fold)
            
            fold_result = {
                'fold': fold + 1,
                'accuracy': accuracy,
                'f1': f1,
                'precision': precision,
                'recall': recall,
                'test_size': len(y_test_fold)
            }
            fold_results.append(fold_result)
            
            # Store for overall analysis
            all_predictions.extend(y_pred_fold)
            all_true_labels.extend(y_test_fold)
            
            print(f"Acc: {accuracy:.4f}, F1: {f1:.4f}, Prec: {precision:.4f}, Rec: {recall:.4f}")
        
        # Calculate overall statistics
        accuracies = [r['accuracy'] for r in fold_results]
        f1_scores = [r['f1'] for r in fold_results]
        precisions = [r['precision'] for r in fold_results]
        recalls = [r['recall'] for r in fold_results]
        
        cv_results = {
            'accuracy': {'mean': np.mean(accuracies), 'std': np.std(accuracies), 'scores': accuracies},
            'f1': {'mean': np.mean(f1_scores), 'std': np.std(f1_scores), 'scores': f1_scores},
            'precision': {'mean': np.mean(precisions), 'std': np.std(precisions), 'scores': precisions},
            'recall': {'mean': np.mean(recalls), 'std': np.std(recalls), 'scores': recalls},
            'fold_results': fold_results
        }
        
        # Print summary
        print(f"Cross-Validation Results:")
        print(f"Accuracy:  {cv_results['accuracy']['mean']:.4f} ± {cv_results['accuracy']['std']:.4f}")
        print(f"F1-Score:  {cv_results['f1']['mean']:.4f} ± {cv_results['f1']['std']:.4f}")
        print(f"Precision: {cv_results['precision']['mean']:.4f} ± {cv_results['precision']['std']:.4f}")
        print(f"Recall:    {cv_results['recall']['mean']:.4f} ± {cv_results['recall']['std']:.4f}")
        
        # Overall confusion matrix
        overall_cm = confusion_matrix(all_true_labels, all_predictions)
        print(f"\nOverall Confusion Matrix:")
        print(overall_cm)
        
        self.cv_results = cv_results
        return cv_results
    
    def train_final_model(self, texts, labels):
        print(f"Training final model...")
        
        # Preprocessing
        processed_texts = self.preprocess_dataset(texts)
        
        # Split train/test for final evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            processed_texts, labels, 
            test_size=0.2, 
            random_state=self.random_state, 
            stratify=labels
        )
        
        # Vectorization
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)
        
        # Training
        self.classifier.fit(X_train_tfidf, y_train)
        
        # Evaluation
        y_pred = self.classifier.predict(X_test_tfidf)
        
        # Calculate metrics
        final_results = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred
        }
        
        print(f"\nFinal Model Results:")
        print(f"Accuracy: {final_results['accuracy']:.4f}")
        print(f"Precision: {final_results['precision']:.4f}")
        print(f"Recall: {final_results['recall']:.4f}")
        print(f"F1-Score: {final_results['f1_score']:.4f}")
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Non-Tossico', 'Tossico']))
        print(f"\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        self.final_results = final_results
        return final_results
    
    def get_important_features(self, n_features=15):
        feature_names = self.vectorizer.get_feature_names_out()
        coef = self.classifier.coef_[0]
        
        # Most important features for toxic class (positive coefficients)
        toxic_indices = np.argsort(coef)[-n_features:][::-1]
        toxic_features = [(feature_names[i], coef[i]) for i in toxic_indices]
        
        # Most important features for non-toxic class (negative coefficients)
        nontoxic_indices = np.argsort(coef)[:n_features]
        nontoxic_features = [(feature_names[i], coef[i]) for i in nontoxic_indices]
        
        print(f"Top {n_features} important features:")
        print(f"Toxic class:")
        for feature, coef in toxic_features:
            print(f"  {feature}: {coef:.4f}")

        print(f"\nNon-Toxic class:")
        for feature, coef in nontoxic_features:
            print(f"  {feature}: {coef:.4f}")
        
        return toxic_features, nontoxic_features

    def save_model(self, model_path=None, vectorizer_path=None):
        model_path = model_path or self.model_path
        vectorizer_path = vectorizer_path or self.vectorizer_path
        
        joblib.dump(self.classifier, model_path)
        joblib.dump(self.vectorizer, vectorizer_path)
        print(f"Saved: {model_path}")
        print(f"Saved: {vectorizer_path}")

if __name__ == "__main__":
    k_folds = 5
    classifier = ToxicityClassifierKFold(k_folds=k_folds)
    csv_path = 'datasets/classification_and_explaination_toxic_conversation_with_non_toxic.csv'
    texts, labels = classifier.load_data(csv_path)
    cv_results = classifier.cross_validate_detailed(texts, labels)
    final_results = classifier.train_final_model(texts, labels)
    classifier.get_important_features()
    classifier.save_model()