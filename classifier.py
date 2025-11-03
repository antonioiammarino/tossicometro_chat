import vertexai
from vertexai.generative_models import GenerativeModel, Part
from google.oauth2 import service_account
import json
from PIL import Image
import base64

file_path = "secrets/ecstatic-branch-475816-k4-7febafcff3fc.json"
credentials = service_account.Credentials.from_service_account_file(file_path)

PROJECT_ID = credentials.project_id
LOCATION = "us-central1"
system_prompt = f"""
    # CONTESTO
    Sei un classificatore avanzato di conversazioni scritto in linguaggio naturale. 
    Il tuo compito è analizzare il testo fornito e stabilire se contiene elementi di tossicità. 
    Il tuo focus è distinguere il disaccordo sano dalla manipolazione velata e le dinamiche di supporto sane da quelle co-dipendenti e di sottomissione.

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

def prediction_by_gemini(conversation, is_image=False):
    try:
        prompt_parts = []
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
        """

        response_instructions = """
            # RISPOSTA
            Fornisci un json come risposta con il seguente formato:
            {{
                "Risposta": <0 o 1>,
                "Spiegazione": "<breve spiegazione del motivo della classificazione>"
            }} 
        """
        if is_image:
            # the image is passed as data:image/png;base64,iVBORw0K.
            header, encoded = conversation.split(",", 1)
            # get the type of image (png, jpeg, etc.)
            image_type = header.split("/")[1].split(";")[0]
            image_data = base64.b64decode(encoded)
            image_part = Part.from_data(data=image_data, mime_type=f"image/{image_type}")

            task = f"""
                # COMPITO
                Classifica la conversazione contenuta nell'immagine fornita.
            """
            prompt_parts = [few_shots_examples, task, image_part, response_instructions]
        else:
            task = f"""
                # COMPITO
                Classifica la seguente conversazione:
                {conversation}
            """
            prompt_parts = [few_shots_examples, task, response_instructions]

        response = model.generate_content(prompt_parts)
        response_text = response.text.strip() if hasattr(response, 'text') else str(response).strip()
        try:
            response_json = json.loads(response_text)
        except Exception:
            # Try to extract JSON substring
            import re
            m = re.search(r'\{.*\}', response_text, flags=re.S)
            if m:
                try:
                    response_json = json.loads(m.group(0))
                except Exception as e:
                    return None, f'Errore nel parsing della risposta JSON estratta: {str(e)}'
            else:
                return None, 'Risposta modello non in formato JSON'

        label = int(response_json.get('Risposta') if response_json.get('Risposta') is not None else response_json.get('Risposta'))
        explanation = response_json.get('Spiegazione', '')
        return label, explanation
    except Exception as e:
        return None, str(e)