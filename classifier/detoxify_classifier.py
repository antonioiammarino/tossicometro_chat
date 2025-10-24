from detoxify import Detoxify
import pandas as pd

# Use the 'multilingual' model for Italian support
model = Detoxify('multilingual')

if __name__ == "__main__":
    for file_name in ['datasets/classification_and_explaination_toxic_conversation(final_normalized_names).csv', 'datasets/classification_and_explaination_toxic_conversation(final_normalized_anon).csv']:
        print(f"Processing file: {file_name}")
        df = pd.read_csv(file_name)
        for _, row in df.iterrows():
            conversation = row.get('conversation', '')
            if conversation:
                result = model.predict(conversation)
                # save in a new column 'toxicity_score'
                df.at[_, 'Score_Tossicita'] = result.get('toxicity', 0.0)
                if result.get('toxicity', 0.0) > 0.5:
                    df.at[_, 'Tossica_per_detoxify'] = 'Sì'
                else:
                    df.at[_, 'Tossica_per_detoxify'] = 'No'
        df.to_csv(f"detoxified_{file_name}", index=False)
        # Calculate and print metrics
        y_true = df['toxic'].tolist()
        y_pred = df['Tossica_per_detoxify'].tolist()
        # Since the toxic column contains only “Sì” (all samples are positive), precision is artificially 
        # equal to 1.0 (no false positives are possible). Consequently, the F1-score is also biased by this 
        # inflated precision. In this scenario, the only meaningful metric is recall, which measures the 
        # proportion of toxic samples correctly identified.
        true_positives = sum(1 for true, pred in zip(y_true, y_pred) if true == 'Sì' and pred == 'Sì')
        false_negatives = sum(1 for true, pred in zip(y_true, y_pred) if true == 'Sì' and pred == 'No')
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        print(f"Recall for {file_name}: {recall:.4f}")