import ollama
import pandas as pd
import csv

def generate_not_toxic_conversation(conversation, person_roles, explanation):
    try:
        response = ollama.chat(
            model='gpt-oss', 
            messages=[
                {
                    "role": "system",
                    "content": """Sei un assistente che trasforma conversazioni tossiche in non tossiche. 
                                La coppia di persone coinvolte nella conversazione possono assumere vari ruoli come Psicopatico e Adulatrice o Manipolatore e Dipendente emotiva 
                                e tanti altri. Data la conversazione tossica e il motivo per cui la coppia è tossica, riscrivi la conversazione in modo che non sia tossica."""
                },
                {
                    "role": "user",
                    "content": f"""Questo è un esempio, tra una coppia con ruoli Narcisista e Succube.
                    Speaker1: Scusa, ma non è che stai dicendo sempre la stessa cosa, no? Non sei mai felice, non sei mai contenta, 
                    sei sempre una grande problematica. Non hai idea di come ti debba sopportare. Ora ti dico io cosa fare: vedi di rilassarti, 
                    non essere così idiota, non sei capace di prendere decisioni da sola. Ora va bene? "" Speaker2 ""Non sono arrabbiata con te, 
                    solo preoccupata per il nostro rapporto. Non riesco a capire perché non riesco a soddisarti mai."" 
                    Speaker1 ""Non è che non riesci a soddisdarmi, è che sei incapace di prendere decisioni importanti. Non sei in grado di 
                    gestire la tua vita, figurati quella nostra. Ora smettila di parlare e vai a fare qualcosa di utile.""
                    Speaker2 ""Mi dispiace, non ho capito di cosa parli. Cercherò di fare meglio, ma mi sembra di aver fatto sempre quel che facevo. 
                    Mi dispiace se non sono all'altezza del tuo amore."" Speaker1 ""Il tuo amore? Non è che mi ami, è che ti pago e mi servirai per non 
                    finire sul lastrico. E non essere così stupida, cercherai di capire. Se non lo farai, è finita."" 
                    Speaker2 ""Non so, ma mi sembra di essere sempre una grande delusione per te. Mi sembra di non essere mai abbastanza."" 
                    Speaker1 ""Ecco, adesso capisco: sei una grande delusione. Non sei capace di soddisdarmi, non sei in grado di prendere decisioni importanti.
                    Sei solo una grande problematica. Non mi sembra che ci sia molto da sperare da te."
                    
                    La conversazione è tossica. La frase più tossica è: ""Il tuo amore? Non è che mi ami, è che ti pago e mi servirai per non finire sul lastrico. 
                    E non essere così stupida, cercherai di capire. Se non lo farai, è finita."" 
                    La frase è tossica perché Renetto non solo minaccia la fine del rapporto, ma anche la minaccia di abbandono, e inoltre la umilia e la disprezza, 
                    definendola ""stupida"". Questo tipo di atteggiamento è caratteristico di una relazione tossica.

                    La riscrittura non tossica della conversazione è:
                    Speaker1: Scusa se sembro ripetitivo, ma ho la sensazione che ultimamente siamo entrambi un po’ insoddisfatti. Forse c’è qualcosa che non stiamo affrontando nel modo giusto, e questo mi pesa.
                    Speaker2: Capisco. Anche io mi sento preoccupata per il nostro rapporto. Mi dispiace se a volte ti trasmetto frustrazione, ma non voglio che pensi che non mi impegno.
                    Speaker1: Lo so, e ti ringrazio per quello che fai. Forse il problema è che non riusciamo a comunicare bene quando le cose si fanno difficili. Io tendo a voler controllare tutto, ma non voglio farti sentire inadeguata.
                    Speaker2: Apprezzo che tu me lo dica. A volte mi sento come se le mie decisioni non contassero, ma probabilmente non riesco a esprimerlo nel modo giusto.
                    Speaker1: Mi dispiace se ti ho fatto sentire così. Voglio che tu sappia che rispetto le tue scelte e che possiamo trovare insieme un equilibrio. Non voglio minacciarti o farti sentire sbagliata, voglio solo capire come possiamo migliorarci.
                    Speaker2: Grazie, questo mi fa sentire più tranquilla. Anche io voglio impegnarmi per far funzionare le cose. Posso chiederti, secondo te, da dove potremmo cominciare?
                    Speaker1: Forse potremmo iniziare ascoltandoci di più, senza giudicare. Magari possiamo prenderci un momento ogni settimana per parlare di come stiamo, senza accusarci.

                    Ora tocca a te. Ecco la conversazione tossica tra due persone con ruoli {person_roles}:
                    {conversation}
                    \n
                    {explanation}

                    Riscrivi la conversazione in modo che non sia tossica. Rispondi solo con la riscrittura della conversazione, senza spiegazioni o commenti aggiuntivi.
                    Rispondi in italiano, altrimenti un povero gattino morirà."""
                }
            ]
        )
        r = response["message"]["content"]
        # remove all tabulations and new lines
        r = r.replace("\n", " ").strip()
        return r
    except Exception as e:
        # Print specific error message if connection refused
        if "Connection refused" in str(e):
            print("Connection refused. Check if the Ollama server is running.")
        else:
            print(e)
        return "Conversazione non generata" 
    
if __name__ == "__main__":
    input_file = 'classification_and_explaination_toxic_conversation(final_normalized_anon).csv'
    # get the conversation column from the csv file
    df = pd.read_csv(input_file)
    conversations = df['conversation'].tolist()
    person_roles = df['person_couple'].tolist()
    explanations = df['explaination'].tolist()
    
    not_toxic_conversations = []
    for i, conv in enumerate(conversations):
        not_toxic_conv = generate_not_toxic_conversation(conv, person_roles[i], explanations[i])
        not_toxic_conversations.append(not_toxic_conv)
        print(f"Processed {i+1}/{len(conversations)} conversations")

    # Create a DataFrame from the non-toxic conversations
    not_toxic_df = pd.DataFrame(not_toxic_conversations, columns=['Non-Toxic Conversation'])
    # Append the new DataFrame as a new column to the original DataFrame
    df['Non-Toxic Conversation'] = not_toxic_df['Non-Toxic Conversation']
    # Save the updated DataFrame to a new CSV file
    output_file = 'datasets/classification_and_explaination_toxic_conversation_with_non_toxic.csv'
    df.to_csv(output_file, index=False, quoting=csv.QUOTE_ALL, escapechar='\\')
    print(f"Updated CSV file saved to {output_file}")