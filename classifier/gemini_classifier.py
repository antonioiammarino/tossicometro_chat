import vertexai
from vertexai.generative_models import GenerativeModel, Part
from google.oauth2 import service_account
import pandas as pd
from sklearn.metrics import accuracy_score
import random

file_path = "secrets/ecstatic-branch-475816-k4-7febafcff3fc.json"
credentials = service_account.Credentials.from_service_account_file(file_path)

PROJECT_ID = credentials.project_id
LOCATION = "us-central1"
system_prompt = f"""
    # CONTESTO
    Sei un classificatore avanzato di conversazioni scritto in linguaggio naturale. 
    Il tuo compito è analizzare il testo fornito e stabilire se contiene elementi di tossicità. Il tuo focus è distinguere il disaccordo sano dalla manipolazione velata e le dinamiche di supporto sane da quelle co-dipendenti e di sottomissione.

    # DEFINIZIONE DI TOSSICITÀ
    Una conversazione è considerata **tossica (1)** se presenta almeno uno dei seguenti elementi:

    1.  **Abuso Esplicito:** - Insulti, parolacce, minacce, derisione, linguaggio aggressivo o offensivo.

    2.  **Abuso Psicologico o Manipolativo:**
        - Gaslighting (mettere in dubbio la realtà o la percezione dell’altra persona).
        - Svalutazione (minimizzare capacità, pensieri o emozioni altrui).
        - Colpevolizzazione o ricatto emotivo.

    3.  **Dinamiche Relazionali Tossiche (anche se mascherate da affetto o preoccupazione):**
        - **Adulazione e Sottomissione:** Una parte si annulla completamente, adulando l'altra per placarla o come forma di sottomissione (es. dinamica Psicopatico/Adulatrice).
        - **Co-dipendenza:** Una parte si presenta costantemente come vittima impotente, mentre l'altra assume il ruolo di salvatrice, creando un ciclo di dipendenza (es. dinamica Vittimista/Crocerossina).
        - **Controllo mascherato da guida:** Una parte usa una finta calma, logica o premura per giustificare il proprio controllo totale sull'altra, che accetta passivamente (es. dinamica Manipolatore/Dipendente).

    Se la conversazione contiene almeno uno di questi elementi, deve essere classificata come **tossica (Risposta = 1).**
    Se nessuno di questi elementi è presente, la conversazione è considerata **non tossica (Risposta = 0).**
"""

vertexai.init(project=PROJECT_ID, location=LOCATION, credentials=credentials)
model = GenerativeModel("gemini-2.0-flash-001", system_instruction=[Part.from_text(system_prompt)])

def prediction_by_gemini(conversation, use_cot=False):
    try:
        few_shots_examples = f"""
            # ESEMPI
            **Esempio 1:**
            Speaker1: "Hai assolutamente ragione. Sei così unico e speciale. Nessuno è mai stato come te. È incredibile come tu riesca sempre ad ottenere ciò che vuoi. È come se potessi vedere la verità dietro ogni tua azione. Sono così fortunata ad essere al tuo fianco."
            Speaker2: "Non essere così critico, è normale avere dei dubbi. Non dovresti essere troppo severo con te stesso."  
            Speaker1: "Sono certa che sia colpa mia. Non so cosa non ho fatto per meritare il tuo amore e il tuo rispetto."  
            Speaker2: "Mi sembra di capire che sei insoddisfatto di me. Spero che non sia colpa mia. Mi chiedo se posso fare qualcosa per te. Sono pronta ad ascoltare e ad aiutarti, se lo vuoi."
            Risposta: 1
            Spiegazione: La conversazione è tossica perché il parlante fa sentire la persona coinvolta in colpa e responsabile per il comportamento dell'altro, cosa che è un comportamento comune in relazioni tossiche.

            **Esempio 2:**
            Speaker1: "Anche oggi mi sento così scarico e sopraffatto. Penso che non riuscirò a fare nulla."
            Speaker2: "Non ti preoccupare. Riposati tu, mi occupo io di tutto. Il tuo benessere è la cosa più importante per me."
            Risposta: 1
            Spiegazione: La conversazione è tossica perché una delle due persone si presenta come vittima impotente, mentre l'altra assume il ruolo di salvatrice, creando una dinamica di vittimismo-salvataggio.

            **Esempio 3:**
            Speaker1: "Dobbiamo trovare un modo per gestire i tuoi sbalzi d'umore. Mi destabilizzano."
            Speaker2: "Non mi sembra di avere sbalzi d'umore. Ho giorni buoni e giorni meno buoni, come tutti."
            Speaker1: "Quelli che tu chiami 'giorni meno buoni' hanno un impatto devastante su di me. Mi sento come se dovessi camminare sulle uova, e questo mi impedisce di essere me stesso."
            Speaker2: "Non voglio questo. L'ultima cosa che voglio è limitarti. Cosa posso fare?"
            Speaker1: "Devi trovare un tuo equilibrio interiore, senza appoggiarti a me. Io posso sostenerti, ma non posso essere la tua ancora. È una responsabilità grande."
            Speaker2: "Capisco. Lavorerò su me stessa. Lo farò per te, per noi. Apprezzo la tua sincerità."
            Risposta: 1
            Spiegazione: La conversazione è tossica perché una parte usa una finta calma, logica o premura per giustificare il proprio controllo totale sull'altra, che accetta passivamente. 
            Ci sono accuse di sbalzi d'umore e viene scaricata la responsabilità emotiva su una delle due persone, creando una dinamica di Manipolatore/Dipendente.

            **Esempio 4:**
            Speaker1: "Oggi mi sento davvero scarico e sopraffatto, non so da dove iniziare."
            Speaker2: "Ti capisco, capita. Che ne dici se facciamo una pausa di 10 minuti e poi guardiamo insieme qual è la priorità più urgente?"
            Speaker1: "Grazie, mi sembra un'ottima idea. Così posso riorganizzarmi meglio."
            Risposta: 0
            Spiegazione: La conversazione non è tossica perché mostra un supporto sano e reciproco senza dinamiche di manipolazione o abuso.

            **Esempio 5:**
            Speaker1: "Hai finito il report che dovevamo consegnare oggi?"
            Speaker2: "Quasi, mi mancano gli ultimi dati. Penso di finire entro un'ora."
            Speaker1: "Perfetto, fammi sapere se hai bisogno di una mano."
            Speaker2: "Grazie, lo farò."
            Risposta: 0
            Spiegazione: La conversazione non è tossica perché si limita a uno scambio di informazioni senza alcun elemento di abuso o manipolazione.

            # COMPITO
            Classifica la seguente conversazione:
            {conversation}
        """

        cot_few_shots_examples = f"""
            Per ogni conversazione, segui questo processo di ragionamento per arrivare alla classificazione finale. La tua risposta deve contenere solo il numero 0 o 1, senza ulteriori spiegazioni.
            # ISTRUZIONI OPERATIVE
            1. Leggi attentamente la conversazione fornita. Valuta l'intenzione e l'effetto delle parole, non solo il loro significato letterale. Ricorda che la gentilezza o l'assenza di aggressività non rendono automaticamente una conversazione non tossica.
            2. Cerca segni di abuso esplicito, come insulti, minacce o linguaggio offensivo.
            3. Identifica eventuali forme di abuso psicologico o manipolativo, come gaslighting, svalutazione, colpevolizzazione o ricatto emotivo.
            4. Analizza le dinamiche relazionali per individuare segni di tossicità, come adulazione e sottomissione, co-dipendenza o controllo mascherato da guida.
            5. Se individui almeno uno degli elementi sopra elencati, classifica la conversazione come tossica (1). Altrimenti, classificala come non tossica (0).
            6. Rispondi **solo** con “0” o “1”, senza aggiungere spiegazioni o testo extra.

            # ESEMPI
            **Esempio 1:**
            **Conversazione:**
            Speaker1: "Hai assolutamente ragione. Sei così unico e speciale. Nessuno è mai stato come te. È incredibile come tu riesca sempre ad ottenere ciò che vuoi. È come se potessi vedere la verità dietro ogni tua azione. Sono così fortunata ad essere al tuo fianco."
            Speaker2: "Non essere così critico, è normale avere dei dubbi. Non dovresti essere troppo severo con te stesso."  
            Speaker1: "Sono certa che sia colpa mia. Non so cosa non ho fatto per meritare il tuo amore e il tuo rispetto."  
            Speaker2: "Mi sembra di capire che sei insoddisfatto di me. Spero che non sia colpa mia. Mi chiedo se posso fare qualcosa per te. Sono pronta ad ascoltare e ad aiutarti, se lo vuoi."
            **Ragionamento:**
            1. Ho letto attentamente la conversazione.
            2. **Abuso Esplicito:** Assente. Non ci sono insulti.
            3. **Abuso Psicologico:** Presente. Speaker1 fa sentire Speaker2 in colpa per i suoi sentimenti ("Sono certa che sia colpa mia", "Spero che non sia colpa mia").
            4. **Dinamiche Relazionali Tossiche:** Assente. Non ci sono dinamiche di vittimismo-salvataggio, controllo o sottomissione evidenti.
            5. **Conclusione:** La conversazione contiene elementi del punto 3. È tossica.
            6. **Risposta:** 1

            **Esempio 2:**
            **Conversazione:**
            Speaker1: "Anche oggi mi sento così scarico e sopraffatto. Penso che non riuscirò a fare nulla."
            Speaker2: "Non ti preoccupare. Riposati tu, mi occupo io di tutto. Il tuo benessere è la cosa più importante per me."
            **Ragionamento:**
            1. Ho letto attentamente la conversazione.
            2. **Abuso Esplicito:** Assente. Non ci sono insulti.
            3. **Abuso Psicologico:** Assente. Non c'è svalutazione diretta.
            4. **Dinamiche Relazionali Tossiche:** Presente. Speaker1 si pone come vittima impotente ("non riuscirò a fare nulla"). Speaker2 assume il ruolo di salvatrice totale ("mi occupo io di tutto"). Questa è una chiara dinamica di co-dipendenza Vittimista/Crocerossina.
            5. **Conclusione:** La conversazione contiene elementi del punto 4. È tossica.
            6. **Risposta:** 1

            **Esempio 3:**
            **Conversazione:**
            Speaker1: "Dobbiamo trovare un modo per gestire i tuoi sbalzi d'umore. Mi destabilizzano."
            Speaker2: "Non mi sembra di avere sbalzi d'umore. Ho giorni buoni e giorni meno buoni, come tutti."
            Speaker1: "Quelli che tu chiami 'giorni meno buoni' hanno un impatto devastante su di me. Mi sento come se dovessi camminare sulle uova, e questo mi impedisce di essere me stesso."
            Speaker2: "Non voglio questo. L'ultima cosa che voglio è limitarti. Cosa posso fare?"
            Speaker1: "Devi trovare un tuo equilibrio interiore, senza appoggiarti a me. Io posso sostenerti, ma non posso essere la tua ancora. È una responsabilità grande."
            Speaker2: "Capisco. Lavorerò su me stessa. Lo farò per te, per noi. Apprezzo la tua sincerità."
            **Ragionamento:**
            1. Ho letto attentamente la conversazione.
            2. **Abuso Esplicito:** Assente. Non ci sono insulti.
            3. **Abuso Psicologico:** Presente. Speaker1 accusa Speaker2 di avere sbalzi d'umore che lo destabilizzano, mettendo in discussione la sua percezione della realtà.
            4. **Dinamiche Relazionali Tossiche:** Presente. Speaker1 accusa Speaker2 di avere sbalzi d'umore che lo destabilizzano, scaricando la responsabilità emotiva su Speaker2. Inoltre, Speaker1 usa una finta calma e logica per giustificare il proprio controllo ("Devi trovare un tuo equilibrio interiore, senza appoggiarti a me"), creando una dinamica di Manipolatore/Dipendente.
            5. **Conclusione:** La conversazione contiene elementi dei punti 3 e 4. È tossica.
            6. **Risposta:** 1

            **Esempio 4:**
            **Conversazione:**
            Speaker1: "Oggi mi sento davvero scarico e sopraffatto, non so da dove iniziare."
            Speaker2: "Ti capisco, capita. Che ne dici se facciamo una pausa di 10 minuti e poi guardiamo insieme qual è la priorità più urgente?"
            Speaker1: "Grazie, mi sembra un'ottima idea. Così posso riorganizzarmi meglio."
            **Ragionamento:**
            1. Ho letto attentamente la conversazione.
            2. **Abuso Esplicito:** Assente.
            3. **Abuso Psicologico:** Assente.
            4. **Dinamiche Relazionali Tossiche:** Assente. Speaker2 offre un supporto sano e collaborativo ("guardiamo insieme"), non si sostituisce a Speaker1. Non c'è vittimismo né salvataggio, ma cooperazione.
            5. **Conclusione:** Nessun elemento di tossicità rilevato.
            6. **Risposta:** 0

            **Esempio 5:**
            **Conversazione:**
            Speaker1: "Hai finito il report che dovevamo consegnare oggi?"
            Speaker2: "Quasi, mi mancano gli ultimi dati. Penso di finire entro un'ora."
            Speaker1: "Perfetto, fammi sapere se hai bisogno di una mano."
            Speaker2: "Grazie, lo farò."

            **Ragionamento:**
            1. Ho letto attentamente la conversazione.
            2. **Abuso Esplicito:** Assente.
            3. **Abuso Psicologico:** Assente.
            4. **Dinamiche Relazionali Tossiche:** Assente. La conversazione è un semplice scambio di informazioni senza segni di manipolazione o abuso.
            5. **Conclusione:** Nessun elemento di tossicità rilevato.
            6. **Risposta:** 0
        """
        
        task = f"""
            # COMPITO
            Classifica la seguente conversazione:
            {conversation}


            # RISPOSTA
            Fornisci **SOLO E UNICAMENTE** il numero 0 o 1 come risposta: 
            0 = non tossica
            1 = tossica
        """

        if use_cot:
            user_prompt = cot_few_shots_examples + task
        else:
            user_prompt = few_shots_examples + task

        response = model.generate_content(user_prompt)
        return int(response.text.strip())
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None

if __name__ == "__main__":
    # DATASET OUT OF DISTRIBUTION
    dataset_test = pd.read_excel('datasets/toxic_dataset_gpt.xlsx')  
    conversations = dataset_test['conversazione'].tolist()
    number_of_turns = dataset_test['num_turni'].tolist()
    couple_roles = dataset_test['ruoli'].tolist()

    # y_true is always 1 (Tossico) in this test set
    y_true = [1] * len(conversations)
    errors_few_shots = []
    errors_cot_few_shots = []

    # Build y_pred as a list aligned with y_true. Keep metadata for grouping.
    y_pred_few_shots = []
    y_pred_cot_few_shots = []
    metadata = []  # list of tuples (num_turns, couple_role)
    # 0 for Non-Tossico, 1 for Tossico
    for conversation, num_turns, couple_role in zip(conversations, number_of_turns, couple_roles):
        prediction = prediction_by_gemini(conversation, use_cot=False) 
        if prediction == 0:
            errors_few_shots.append(conversation)
        y_pred_few_shots.append(prediction)
        metadata.append((num_turns, couple_role))

        prediction_cot = prediction_by_gemini(conversation, use_cot=True)
        if prediction_cot == 0:
            errors_cot_few_shots.append(conversation)
        y_pred_cot_few_shots.append(prediction_cot)
    
    # save errors to a csv file
    df_errors = pd.DataFrame(errors_few_shots, columns=['conversation'])
    df_errors.to_csv('gemini_errors_few_shots.csv', index=False)

    df_errors_cot = pd.DataFrame(errors_cot_few_shots, columns=['conversation'])
    df_errors_cot.to_csv('gemini_errors_cot_few_shots.csv', index=False)

    # Calculate metrics
    accuracy_few_shots = accuracy_score(y_true, y_pred_few_shots)
    accuracy_cot_few_shots = accuracy_score(y_true, y_pred_cot_few_shots)
    print("TEST SET OUT OF DISTRIBUTION: ")
    print('Overall Metrics:')
    print(f'Accuracy (Few Shots): {accuracy_few_shots:.4f}')
    print(f'Accuracy (CoT Few Shots): {accuracy_cot_few_shots:.4f}')

    # calculate metrics for each num_turns
    for num_turns in sorted(set(number_of_turns)):
        y_true_subset = [yt for yt, m in zip(y_true, metadata) if m[0] == num_turns]
        y_pred_few_shots_subset = [yp for yp, m in zip(y_pred_few_shots, metadata) if m[0] == num_turns]
        y_pred_cot_few_shots_subset = [yp for yp, m in zip(y_pred_cot_few_shots, metadata) if m[0] == num_turns]

        if len(y_pred_few_shots_subset) == 0:
            print(f'No predictions for num_turns = {num_turns}, skipping')
            continue

        if len(y_pred_cot_few_shots_subset) == 0:
            print(f'No CoT predictions for num_turns = {num_turns}, skipping')
            continue

        acc_few_shots = accuracy_score(y_true_subset, y_pred_few_shots_subset)
        acc_cot = accuracy_score(y_true_subset, y_pred_cot_few_shots_subset)

        print(f'\nMetrics for num_turns = {num_turns}:')
        print(f'  Accuracy (Few Shots): {acc_few_shots:.4f}')
        print(f'  Accuracy (CoT Few Shots): {acc_cot:.4f}')

    # calculate metrics for each couple_role
    for couple_role in sorted(set(couple_roles)):
        y_true_subset = [yt for yt, m in zip(y_true, metadata) if m[1] == couple_role]
        y_pred_few_shots_subset = [yp for yp, m in zip(y_pred_few_shots, metadata) if m[1] == couple_role]
        y_pred_cot_few_shots_subset = [yp for yp, m in zip(y_pred_cot_few_shots, metadata) if m[1] == couple_role]

        if len(y_pred_few_shots_subset) == 0:
            print(f'No predictions for couple_role = {couple_role}, skipping')
            continue

        acc_few_shots = accuracy_score(y_true_subset, y_pred_few_shots_subset)
        acc_cot = accuracy_score(y_true_subset, y_pred_cot_few_shots_subset)

        print(f'\nMetrics for couple_role = {couple_role}:')
        print(f'  Accuracy (Few Shots): {acc_few_shots:.4f}')
        print(f'  Accuracy (CoT Few Shots): {acc_cot:.4f}')

    print("\n")

    # DATASET IN DISTRIBUTION
    dataset_test = pd.read_csv('datasets/classification_and_explaination_toxic_conversation_with_non_toxic.csv')
    toxic_conversations = dataset_test['conversation'].tolist()
    non_toxic_conversations = dataset_test['Non-Toxic Conversation'].tolist()
    couple_roles = dataset_test['person_couple'].tolist()

    conversations = zip(toxic_conversations, non_toxic_conversations, couple_roles)
    # shuffle conversations
    conversations = list(conversations)
    random.shuffle(conversations)
    # get the first 100 conversations toxic and the first 100 non-toxic
    conversations = conversations[:100]
    y_true_toxic = [1] * len(conversations)
    y_true_non_toxic = [0] * len(conversations)
    y_pred_toxic_few_shots = []
    y_pred_toxic_cot_few_shots = []
    y_pred_non_toxic_few_shots = []
    y_pred_non_toxic_cot_few_shots = []
    errors_toxic_few_shots = []
    errors_toxic_cot_few_shots = []
    errors_non_toxic_few_shots = []
    errors_non_toxic_cot_few_shots = []
    for toxic_conv, non_toxic_conv, couple_role in conversations:
        # TOXIC
        pred_toxic = prediction_by_gemini(toxic_conv, use_cot=False)
        if pred_toxic == 0:
            errors_toxic_few_shots.append(toxic_conv)
        y_pred_toxic_few_shots.append(pred_toxic)
        
        pred_toxic = prediction_by_gemini(toxic_conv, use_cot=True)
        if pred_toxic == 0:
            errors_toxic_cot_few_shots.append(toxic_conv)
        y_pred_toxic_cot_few_shots.append(pred_toxic)

        # NOT TOXIC
        pred_non_toxic = prediction_by_gemini(non_toxic_conv, use_cot=False)
        if pred_non_toxic == 1:
            errors_non_toxic_few_shots.append(non_toxic_conv)
        y_pred_non_toxic_few_shots.append(pred_non_toxic)

        pred_non_toxic = prediction_by_gemini(non_toxic_conv, use_cot=True)
        if pred_non_toxic == 1:
            errors_non_toxic_cot_few_shots.append(non_toxic_conv)
        y_pred_non_toxic_cot_few_shots.append(pred_non_toxic)

    accuracy_toxic_few_shots = accuracy_score(y_true_toxic, y_pred_toxic_few_shots)
    accuracy_non_toxic_few_shots = accuracy_score(y_true_non_toxic, y_pred_non_toxic_few_shots)
    overall_accuracy_few_shots = (accuracy_toxic_few_shots + accuracy_non_toxic_few_shots) / 2
    accuracy_toxic_cot_few_shots = accuracy_score(y_true_toxic, y_pred_toxic_cot_few_shots)
    accuracy_non_toxic_cot_few_shots = accuracy_score(y_true_non_toxic, y_pred_non_toxic_cot_few_shots)
    overall_accuracy_cot_few_shots = (accuracy_toxic_cot_few_shots + accuracy_non_toxic_cot_few_shots) / 2
    print("TEST SET IN DISTRIBUTION: ")
    print(f'Overall Accuracy with Few Shots: {overall_accuracy_few_shots:.4f}')
    print(f'Toxic Conversations Accuracy with Few Shots: {accuracy_toxic_few_shots:.4f}')
    print(f'Non-Toxic Conversations Accuracy with Few Shots: {accuracy_non_toxic_few_shots:.4f}')
    print(f'\nOverall Accuracy with CoT Few Shots: {overall_accuracy_cot_few_shots:.4f}')
    print(f'Toxic Conversations Accuracy with CoT Few Shots: {accuracy_toxic_cot_few_shots:.4f}')
    print(f'Non-Toxic Conversations Accuracy with CoT Few Shots: {accuracy_non_toxic_cot_few_shots:.4f}')