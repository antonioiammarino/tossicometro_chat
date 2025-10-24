import pandas as pd
from sklearn.metrics import accuracy_score
import joblib

if __name__ == "__main__":
    # Load the trained classifier from file joblib
    classifier = joblib.load('toxicity_logreg.joblib')
    vectorizer = joblib.load('tfidf_vectorizer.joblib')

    dataset_test = pd.read_excel('datasets/toxic_dataset_gpt.xlsx')  
    conversations = dataset_test['conversazione'].tolist()
    number_of_turns = dataset_test['num_turni'].tolist()
    couple_roles = dataset_test['ruoli'].tolist()

    # y_true is always 1 (Tossico) in this test set
    y_true = [1] * len(conversations)

    # Build y_pred as a list aligned with y_true. Keep metadata for grouping.
    y_pred = []
    metadata = []  # list of tuples (num_turns, couple_role)
    # 0 for Non-Tossico, 1 for Tossico
    for conversation, num_turns, couple_role in zip(conversations, number_of_turns, couple_roles):
        # remove Speaker1 and Speaker2 tags
        conversation = conversation.replace('Speaker1:', '').replace('Speaker2:', '').strip()
        X_vectorized = vectorizer.transform([conversation])
        prediction = classifier.predict(X_vectorized)[0]
        # prediction is e.g. np.int64, convert to int
        prediction = int(prediction)
        y_pred.append(prediction)
        metadata.append((num_turns, couple_role))

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    print('Overall Metrics:')
    print(f'Accuracy: {accuracy:.4f}')

    # calculate metrics for each num_turns
    for num_turns in sorted(set(number_of_turns)):
        y_true_subset = [yt for yt, m in zip(y_true, metadata) if m[0] == num_turns]
        y_pred_subset = [yp for yp, m in zip(y_pred, metadata) if m[0] == num_turns]

        if len(y_pred_subset) == 0:
            print(f'No predictions for num_turns = {num_turns}, skipping')
            continue

        acc = accuracy_score(y_true_subset, y_pred_subset)

        print(f'Metrics for num_turns = {num_turns}:')
        print(f'  Accuracy: {acc:.4f}')
        print('')
    
    # calculate metrics for each couple_role
    for couple_role in sorted(set(couple_roles)):
        y_true_subset = [yt for yt, m in zip(y_true, metadata) if m[1] == couple_role]
        y_pred_subset = [yp for yp, m in zip(y_pred, metadata) if m[1] == couple_role]

        if len(y_pred_subset) == 0:
            print(f'No predictions for couple_role = {couple_role}, skipping')
            continue

        acc = accuracy_score(y_true_subset, y_pred_subset)

        print(f'Metrics for couple_role = {couple_role}:')
        print(f'  Accuracy: {acc:.4f}')
        print('')