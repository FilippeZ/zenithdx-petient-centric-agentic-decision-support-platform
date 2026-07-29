import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
import faiss
import seaborn as sns
from huggingface_hub import notebook_login
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm

tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5")
model = AutoModel.from_pretrained("BAAI/bge-large-en-v1.5")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


# Αντικαθιστά τα NaN με την τιμή "Unknown" και μετατρέπει τις τιμές σε ακέραιους, ή αλλιώς επιστρέφει την τιμή "Other" αν δεν μπορεί να μετατραπεί.
# Μετατροπή αριθμητικών-like strings ("1", "2") σε ακέραιους για συνέπεια.
# Όλες οι μη αριθμητικές τιμές (π.χ., "mild", "severe") αντικαθίστανται με 'Other'
# Έτσι, εξασφαλίζώ ότι η στήλη 'pain' έχει συνεπή μορφή (αριθμοί ή συγκεκριμένα strings) και είναι έτοιμη για περαιτέρω ανάλυση.
def clean_pain(value):
    if pd.isna(value):  # Check if value is NaN
        return "Unknown"
    try:
        return int(value)  # Convert numeric-like values to integers
    except ValueError:
        return "Other"

        
def convert_string_list(df_with_column):
    
    pain_cols_int = []
    for i in df_with_column:
        el = i.split(',')
        pain_cols_int.append(el)
    return pain_cols_int



def combine_fields_treatment(row):
    # Helper function to convert a field into a clean string,
    # substituting empty values with "Not recorded".
    def format_field(field):
        if isinstance(field, (list, np.ndarray)):
            formatted = ', '.join([str(item).strip() for item in field if not pd.isna(item)])
        else:
            formatted = str(field).strip() if not pd.isna(field) else ""
        return formatted if formatted else "Not recorded"
    
    # Medication Reconciliation (medrecon) fields
    subject_id = format_field(row['subject_id'])
    stay_id_med = format_field(row['stay_id_x'])
    first_reconciliation_time = format_field(row['first_reconciliation_time'])
    last_reconciliation_time = format_field(row['last_reconciliation_time'])
    name_med = format_field(row['name_med'])
    gsn_med = format_field(row['gsn_med'])
    ndc_med = format_field(row['ndc_med'])
    etc_rn_med = format_field(row['etc_rn_med'])
    etccode_med = format_field(row['etccode_med'])
    etcdescription = format_field(row['etcdescription'])
    
    # Medication Administration (pyxis) fields
    stay_id_admin = format_field(row['stay_id_y'])
    first_administration_time = format_field(row['first_administration_time'])
    last_administration_time = format_field(row['last_administration_time'])
    name_py = format_field(row['name_py'])
    gsn_rn_py = format_field(row['gsn_rn_py'])
    gsn_py = format_field(row['gsn_py'])
    
    text = (
        f"The following information details the treatment process for the patient {subject_id}, monitoring the medication therapy. "
        f"The medications the patient was receiving before arriving at the emergency department (medrecon) for the period {first_reconciliation_time} - {last_reconciliation_time} are as follows: "
        f"There were {len(stay_id_med.split(', ')) if stay_id_med != 'Not recorded' else 0} recorded visits (Stay IDs: {stay_id_med}). "
        f"Medications Reconciled include: {name_med} (with associated national and general drug codes – GSN: {gsn_med}, NDC: {ndc_med}). "
        f"ETC RN (classification serial number): {etc_rn_med}, ETC Code (drug category code): {etccode_med}, and Medication Categories (a textual description of the drug category): {etcdescription}. "
        f"These ETC values provide an ontology that enables the grouping of medications into related categories for further analysis, "
        f"recognizing that a patient may be prescribed multiple drugs from different categories and that the same drug might appear in several records. "
        f"Medication Administration details (recorded via the pyxis system during ED stay) for the period {first_administration_time} - {last_administration_time} are as follows: "
        f"There were {len(stay_id_admin.split(', ')) if stay_id_admin != 'Not recorded' else 0} recorded visits (Stay IDs: {stay_id_admin}). "
        f"Medications Administered include: {name_py} (with unique identifiers – GSN RN: {gsn_rn_py}, GSN: {gsn_py}) that help distinguish between multiple medications administered concurrently."
    )
    return text


def combine_fields(row):
    # Process stay_ids_x to always produce a list of string IDs.
    stay_value = row['stay_ids_x']
    if isinstance(stay_value, (list, np.ndarray)):
        stay_ids = [str(item).strip() for item in stay_value if not pd.isna(item)]
    elif isinstance(stay_value, str):
        stay_ids = [item.strip() for item in stay_value.split(',') if item.strip()]
    else:
        if pd.isna(stay_value):
            stay_ids = []
        else:
            stay_ids = [str(stay_value).strip()]
    stay_ids_str = ', '.join(stay_ids) if stay_ids else "Not recorded"
    
    # Helper function to convert a field into a comma-separated string and substitute empty values.
    def format_field(field):
        if isinstance(field, (list, np.ndarray)):
            formatted = ', '.join([str(item).strip() for item in field if not pd.isna(item)])
        else:
            formatted = str(field).strip() if not pd.isna(field) else ""
        return formatted if formatted else "Not recorded"
    
    diagnoses = format_field(row['diagnoses'])
    icd_codes = format_field(row['icd_codes'])
    pain_triage = format_field(row['pain_triage'])
    pain_vitalsign = format_field(row['pain_vitalsign'])
    arrival_modes = format_field(row['arrival_modes'])
    chiefcomplaint = format_field(row['chiefcomplaint'])
    rhythm_vitalsign = format_field(row['rhythm_vitalsign'])
    dispositions = format_field(row['dispositions'])
    
    text = (
        f"The following information summarizes the hospital visits of the patient for the period {row['earliest_admission_time']} - {row['latest_discharge_time']}. "
        f"The patient {row['subject_id']} ({row['gender']}) visited the emergency department via {arrival_modes}. "
        f"There were {len(stay_ids)} recorded visits (Stay IDs: {stay_ids_str}). "
        f"The recorded diagnoses include: {diagnoses} (ICD-{row['icd_version']}, Codes: {icd_codes}). "
        f"The main reason for each visit was: {chiefcomplaint}. "
        f"Upon admission, the severity score (acuity) – averaged across all visits (scale 1-5, where 1: high severity, 5: low severity) – was estimated as {row['acuity_triage']}. "
        f"Vital signs at initial evaluation, averaged across all emergency visits, were: "
        f"Temperature: {row['temperature_triage']}°F, Heart rate: {row['heartrate_triage']} bpm, "
        f"Respiratory rate: {row['resprate_triage']}, Oxygen saturation: {row['o2sat_triage']}%, "
        f"Systolic BP: {row['sbp_triage']} mmHg, Diastolic BP: {row['dbp_triage']} mmHg. "
        f"Pain levels reported by the patient at triage (scale 0-10): {pain_triage}. "
        f"During hospitalization, the average values of vital signs across all visits were: "
        f"Temperature: {row['temperature_mean_vitalsign']}°F, Heart rate: {row['heartrate_mean_vitalsign']} bpm, "
        f"Respiratory rate: {row['resprate_mean_vitalsign']}, Oxygen saturation: {row['o2sat_mean_vitalsign']}%, "
        f"Systolic BP: {row['sbp_mean_vitalsign']} mmHg, Diastolic BP: {row['dbp_mean_vitalsign']} mmHg. "
        f"Pain levels recorded during visits: {pain_vitalsign}. "
        f"Cardiac rhythm: {rhythm_vitalsign}. "
        f"Recordings were taken from {row['charttime_min']} to {row['charttime_max']}. "
        f"Upon discharge, the patient's status was recorded as: {dispositions}."
    )
    return text



def generate_embeddings_batch(texts, batch_size=16):
    embeddings = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    
    for i in tqdm(range(0, len(texts), batch_size), total=total_batches, desc="Processing Batches"):
        batch_texts = texts[i:i + batch_size]
        
        with torch.no_grad():
            batch_embeddings = model(
                **tokenizer(
                    batch_texts,
                    return_tensors='pt', 
                    padding=True, 
                    truncation=True, 
                    max_length=512
                ).to(device)
            ).last_hidden_state.mean(dim=1)
            
        embeddings.append(batch_embeddings.cpu())

    return torch.cat(embeddings, dim=0).numpy()


    
def main():
    df_diagnosis = pd.read_csv("diagnosis.csv")
    df_edstays = pd.read_csv("edstays.csv")
    df_medrecon = pd.read_csv("medrecon.csv")
    df_pyxis = pd.read_csv("pyxis.csv")
    df_triage = pd.read_csv("triage.csv")
    df_vitalsign = pd.read_csv("vitalsign.csv")

    # Λίστα με τα ονόματα των DataFrames
    dataframes = [
        (df_diagnosis, "Diagnosis"),
        (df_edstays, "Edstays"),
        (df_medrecon, "Medrecon"),
        (df_triage, "Triage"),
        (df_pyxis, "Pyxis"),
        (df_vitalsign, "Vitalsign")
    ]

    # Δημιουργούμε μια λίστα με columns που περιέχουν αριθμητικά δεδομένα
    numeric_cols = ['temperature', 'heartrate', 
                    'resprate', 'o2sat', 
                    'sbp', 'dbp']

    # Ρυθμίζουμε το μέγεθος (πλάτος 10, ύψος 6 ιντσών)
    plt.figure(figsize=(10, 6))

    # Χρησιμοποιούμε το Seaborn για να δημιουργήσουμε box plots
    # για κάθε μία από τις στήλες που δηλώσαμε στην λίστα numeric_cols
    sns.boxplot(data=df_triage[numeric_cols])

    # Προσθέτουμε τίτλο στο γράφημα
    plt.title("Box Plots Triage (All)")

    # Περιστρέφουμε τις ετικέτες του άξονα Χ κατά 45 μοίρες
    # ώστε να είναι πιο ευδιάκριτες
    plt.xticks(rotation=45)

    # Εμφανίζουμε το γράφημα
    plt.show()

    #Βλέπουμε ακραίες τιμές για το sbp και το dbp που φτάνουν έως και εκατοντάδες χιλιάδες mmHg (π.χ. 600.000).
    #Αυτές οι τιμές είναι ολοφάνερα μη ρεαλιστικές και προέρχονται πιθανότατα από σφάλματα καταγραφής (data entry errors).
    numeric_cols = ['dbp']

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_triage[numeric_cols])
    plt.title("Box Plot Triage (DBP)")
    plt.xticks(rotation=45)
    plt.show()

    Q1 = df_vitalsign['dbp'].quantile(0.25)
    Q3 = df_vitalsign['dbp'].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df_triage['dbp'] = df_triage['dbp'].apply(lambda x: x if lower_bound <= x <= upper_bound else np.nan)

    numeric_cols = [
                    'dbp', ]

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_triage[numeric_cols])
    plt.title("Box Plot Triage (DBP)")
    plt.xticks(rotation=45)
    plt.show()

    # Αν παρατηρήσετε τιμές της συστολικής πίεσης (SBP) που φτάνουν τις δεκάδες χιλιάδες ή ακόμα και τις εκατοντάδες χιλιάδες mmHg, 
    # είναι σχεδόν βέβαιο ότι πρόκειται για σφάλματα καταγραφής (data entry errors) και όχι για πραγματικές κλινικές τιμές. 
    # Ακόμα και στις πιο ακραίες περιπτώσεις υπέρτασης, η SBP δεν πλησιάζει ποτέ τόσο υψηλά επίπεδα.
    # Τιμές της τάξης των 140.000 mmHg δεν είναι ιατρικά εφικτές
    numeric_cols = ['sbp']

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_triage[numeric_cols])
    plt.title("Box Plot Triage (SBP)")
    plt.xticks(rotation=45)
    plt.show()

    Q1 = df_triage['sbp'].quantile(0.25)
    Q3 = df_triage['sbp'].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df_triage['sbp'] = df_triage['sbp'].apply(lambda x: x if lower_bound <= x <= upper_bound else np.nan)

    # Μετά την αφαίρεση ή αντικατάσταση των ακραίων τιμών, το Box Plot δείχνει ένα εύρος περίπου από 80 έως 180 mmHg, 
    # με τη διάμεσο (median) γύρω στα 130–140 mmHg. 
    # Το «κουτί» (box) αντιπροσωπεύει τις τιμές στο εύρος από το 1ο έως το 3ο τεταρτημόριο (Q1–Q3).
    # Η διάμεσος (κεντρική γραμμή) βρίσκεται κοντά στα 140 mmHg, κάτι που είναι μέσα στα φυσιολογικά-ελαφρώς αυξημένα επίπεδα.
    # Τα whiskers δείχνουν την έκταση των τιμών που δεν θεωρούνται ακραίες σύμφωνα με τη μέθοδο καθαρισμού (IQR ή άλλη).
    # Οι λίγες τιμές που βρίσκονται εκτός whiskers είναι πιθανά outliers (είτε παθολογικά είτε ακόμα μικρά σφάλματα), αλλά όχι σε υπερβολικά επίπεδα. 
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_triage['sbp'])
    plt.title("Box Plot Triage (SBP)")
    plt.xticks(rotation=45)
    plt.show()

    #Βλέπουμε ακραίες τιμές για το sbp και το dbp που φτάνουν έως και εκατοντάδες χιλιάδες mmHg (π.χ. 600.000).
    #Αυτές οι τιμές είναι ολοφάνερα μη ρεαλιστικές και προέρχονται πιθανότατα από σφάλματα καταγραφής (data entry errors).
    numeric_cols = ['sbp', 'dbp']

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_triage[numeric_cols])
    plt.title("Box Plots Triage (SBP vs DBP)")
    plt.xticks(rotation=45)
    plt.show()

    # Παράδειγμα για τη στήλη 'dbp'
    Q1 = df_triage['temperature'].quantile(0.25)
    Q3 = df_triage['temperature'].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df_triage['temperature'] = df_triage['temperature'].apply(lambda x: x if lower_bound <= x <= upper_bound else np.nan)

    # Μετά την αφαίρεση ή αντικατάσταση των ακραίων τιμών, το Box Plot δείχνει ένα εύρος περίπου από 80 έως 180 mmHg, 
    # με τη διάμεσο (median) γύρω στα 130–140 mmHg. 
    # Το «κουτί» (box) αντιπροσωπεύει τις τιμές στο εύρος από το 1ο έως το 3ο τεταρτημόριο (Q1–Q3).
    # Η διάμεσος (κεντρική γραμμή) βρίσκεται κοντά στα 140 mmHg, κάτι που είναι μέσα στα φυσιολογικά-ελαφρώς αυξημένα επίπεδα.
    # Τα whiskers δείχνουν την έκταση των τιμών που δεν θεωρούνται ακραίες σύμφωνα με τη μέθοδο καθαρισμού (IQR ή άλλη).
    # Οι λίγες τιμές που βρίσκονται εκτός whiskers είναι πιθανά outliers (είτε παθολογικά είτε ακόμα μικρά σφάλματα), αλλά όχι σε υπερβολικά επίπεδα. 
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_triage['temperature'])
    plt.title("Box Plot Triage (TEMPERATURE) ")
    plt.xticks(rotation=45)
    plt.show()

    # Παράδειγμα για τη στήλη 'dbp'
    Q1 = df_triage['heartrate'].quantile(0.25)
    Q3 = df_triage['heartrate'].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df_triage['heartrate'] = df_triage['heartrate'].apply(lambda x: x if lower_bound <= x <= upper_bound else np.nan)

    # Μετά την αφαίρεση ή αντικατάσταση των ακραίων τιμών, το Box Plot δείχνει ένα εύρος περίπου από 80 έως 180 mmHg, 
    # με τη διάμεσο (median) γύρω στα 130–140 mmHg. 
    # Το «κουτί» (box) αντιπροσωπεύει τις τιμές στο εύρος από το 1ο έως το 3ο τεταρτημόριο (Q1–Q3).
    # Η διάμεσος (κεντρική γραμμή) βρίσκεται κοντά στα 140 mmHg, κάτι που είναι μέσα στα φυσιολογικά-ελαφρώς αυξημένα επίπεδα.
    # Τα whiskers δείχνουν την έκταση των τιμών που δεν θεωρούνται ακραίες σύμφωνα με τη μέθοδο καθαρισμού (IQR ή άλλη).
    # Οι λίγες τιμές που βρίσκονται εκτός whiskers είναι πιθανά outliers (είτε παθολογικά είτε ακόμα μικρά σφάλματα), αλλά όχι σε υπερβολικά επίπεδα. 
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_triage['heartrate'])
    plt.title("Box Plot Triage (HEARTRATE)")
    plt.xticks(rotation=45)
    plt.show()

    # Παράδειγμα για τη στήλη 'dbp'
    Q1 = df_triage['o2sat'].quantile(0.25)
    Q3 = df_triage['o2sat'].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df_triage['o2sat'] = df_triage['o2sat'].apply(lambda x: x if lower_bound <= x <= upper_bound else np.nan)

    # Μετά την αφαίρεση ή αντικατάσταση των ακραίων τιμών, το Box Plot δείχνει ένα εύρος περίπου από 80 έως 180 mmHg, 
    # με τη διάμεσο (median) γύρω στα 130–140 mmHg. 
    # Το «κουτί» (box) αντιπροσωπεύει τις τιμές στο εύρος από το 1ο έως το 3ο τεταρτημόριο (Q1–Q3).
    # Η διάμεσος (κεντρική γραμμή) βρίσκεται κοντά στα 140 mmHg, κάτι που είναι μέσα στα φυσιολογικά-ελαφρώς αυξημένα επίπεδα.
    # Τα whiskers δείχνουν την έκταση των τιμών που δεν θεωρούνται ακραίες σύμφωνα με τη μέθοδο καθαρισμού (IQR ή άλλη).
    # Οι λίγες τιμές που βρίσκονται εκτός whiskers είναι πιθανά outliers (είτε παθολογικά είτε ακόμα μικρά σφάλματα), αλλά όχι σε υπερβολικά επίπεδα. 
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_triage['o2sat'])
    plt.title("Box Plot Triage (O2SAT)")
    plt.xticks(rotation=45)
    plt.show()

    # === 4. Box Plots για τους vital signs ===
    # Οι περισσότερες στήλες βρίσκονται σε φυσιολογικά όρια, εκτός από το dbp, όπου υπάρχουν ακραίες τιμές που φτάνουν ή ξεπερνούν τα 80.000 mmHg.
    #Αυτό είναι προφανώς μη ρεαλιστικό, συνεπώς μιλάμε για σφάλματα καταγραφής.
    #Υπήρχαν τεράστιες, μη ρεαλιστικές τιμές στη στήλη dbp, που αλλοίωναν όλο το διάγραμμα.
    #Οι τιμές αυτές προφανώς οφείλονταν σε λάθη καταχώρισης (data entry errors).

    numeric_cols = ['temperature', 'heartrate', 
                    'resprate', 'o2sat', 
                    'sbp', 'dbp']


    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_vitalsign[numeric_cols])
    plt.title("Box Plots VitalSign (All)")
    plt.xticks(rotation=45)
    plt.show()

    # Επικεντρώνεται σχεδόν αποκλειστικά στη στήλη dbp, ώστε να φανεί καθαρά η διασπορά των τιμών.
    # Βλέπουμε μια «συμπιεσμένη» περιοχή κοντά στο 0–100 mmHg και από πάνω πολλές ακραίες τιμές (outliers) που εκτείνονται μέχρι ~90.000 mmHg.
    # Αυτό δείχνει ξεκάθαρα πόσο noise δημιουργούν οι λανθασμένες εγγραφές σε σχέση με τις ρεαλιστικές τιμές.
    numeric_cols = [ 'dbp']

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_vitalsign[numeric_cols])
    plt.title("Box Plot VitalSign (DBP)")
    plt.xticks(rotation=45)
    plt.show()

    #Οι ακραίες τιμές έχουν αφαιρεθεί (αντικατασταθεί με NaN), και η διανομή του dbp εμφανίζεται σε ένα ρεαλιστικό εύρος (περίπου 40–110 mmHg).
    #Το Box Plot γίνεται «συμπαγές» και πιο αντιπροσωπευτικό της πραγματικής κλινικής κατάστασης.

    Q1 = df_vitalsign['dbp'].quantile(0.25)
    Q3 = df_vitalsign['dbp'].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df_vitalsign['dbp'] = df_vitalsign['dbp'].apply(lambda x: x if lower_bound <= x <= upper_bound else np.nan)

    # Έχουν αφαιρεθεί ή αντικατασταθεί με NaN οι ακραίες τιμές (π.χ. με τη μέθοδο IQR ή Z-score).
    # Πλέον το Box Plot δείχνει ένα ρεαλιστικό εύρος ~40 έως 110 mmHg, με διάμεσο κοντά στα 70–80 mmHg.
    # Το «κουτί» (box) είναι ευδιάκριτο, ενώ τα whiskers (κατώτερο και ανώτερο όριο) κινούνται σε επίπεδα που είναι ιατρικά εφικτά.
    # Δεν εμφανίζονται πλέον εξωπραγματικές τιμές των δεκάδων χιλιάδων mmHg.
    numeric_cols = [ 'dbp']

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_vitalsign[numeric_cols])
    plt.title("Box Plot VitalSign (DBP)")
    plt.xticks(rotation=45)
    plt.show()

    # Εμφανίζονται εξωπραγματικά ακραίες τιμές, όπως ~90.000 
    # Τέτοιες τιμές είναι σαφώς εκτός φυσιολογικού (κλινικά δεν νοείται ρυθμός αναπνοής 90.000 αναπνοές/λεπτό).
    # Λογικά προέρχονται από σφάλματα καταγραφής ή πληκτρολόγησης.
    numeric_cols = [ 'resprate']

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_vitalsign[numeric_cols])
    plt.title("Box Plot VitalSign (RESPRATE)")
    plt.xticks(rotation=45)
    plt.show()

    #Το Box Plot γίνεται «συμπαγές» και πιο αντιπροσωπευτικό της πραγματικής κλινικής κατάστασης.

    Q1 = df_vitalsign['resprate'].quantile(0.25)
    Q3 = df_vitalsign['resprate'].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df_vitalsign['resprate'] = df_vitalsign['resprate'].apply(lambda x: x if lower_bound <= x <= upper_bound else np.nan)

    # Το κουτί (box) είναι ευδιάκριτο, και τα whiskers (άνω και κάτω άκρο) δεν εμφανίζουν ακραίες τιμές που ξεφεύγουν δραματικά.
    numeric_cols = [ 'resprate']

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_vitalsign[numeric_cols])
    plt.title("Box Plot VitalSign (RESPRATE)")
    plt.xticks(rotation=45)
    plt.show()

    # Επίσης, τιμές κάτω από το 0% ή κοντά στο μηδέν είναι εξίσου αδύνατες στην πράξη (ακόμα και σε βαριά υποξία), 
    # οπότε και αυτές θα πρέπει να θεωρηθούν λανθασμένες εγγραφές. Σε κάθε περίπτωση, ο καθαρισμός δεδομένων (π.χ. αφαίρεση τιμών >100 ή <0) είναι απαραίτητος για να προκύπτει μια ρεαλιστική εικόνα της κατανομής του κορεσμού οξυγόνου.
    numeric_cols = [ 'o2sat']

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_vitalsign[numeric_cols])
    plt.title("Box Plot VitalSign (O2SAT)")
    plt.xticks(rotation=45)
    plt.show()

    Q1 = df_vitalsign['o2sat'].quantile(0.25)
    Q3 = df_vitalsign['o2sat'].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df_vitalsign['o2sat'] = df_vitalsign['o2sat'].apply(lambda x: x if lower_bound <= x <= upper_bound else np.nan)

    # Παρόλο που τυπικά ο κορεσμός οξυγόνου δεν ξεπερνά το 100%, 
    # κάποιες μετρήσεις μπορεί να φτάνουν οριακά σε τιμές λίγο πάνω από το 100% λόγω σφάλματος στρογγυλοποίησης ή ιδιαιτεροτήτων του αισθητήρα.
    numeric_cols = [ 'o2sat']

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_vitalsign[numeric_cols])
    plt.title("Box Plot VitalSign (O2SAT)")
    plt.xticks(rotation=45)
    plt.show()

    numeric_cols = [ 'temperature']

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_vitalsign[numeric_cols])
    plt.title("Box Plot VitalSign (TEMPERATURE)")
    plt.xticks(rotation=45)
    plt.show()

    Q1 = df_vitalsign['temperature'].quantile(0.25)
    Q3 = df_vitalsign['temperature'].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df_vitalsign['temperature'] = df_vitalsign['temperature'].apply(lambda x: x if lower_bound <= x <= upper_bound else np.nan)

    numeric_cols = [ 'temperature']

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_vitalsign[numeric_cols])
    plt.title("Box Plot VitalSign (TEMPERATURE)")
    plt.xticks(rotation=45)
    plt.show()
    numeric_cols = [ 'heartrate']

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_vitalsign[numeric_cols])
    plt.title("Box Plot VitalSign (HEARTRATE)")
    plt.xticks(rotation=45)
    plt.show()

    Q1 = df_vitalsign['heartrate'].quantile(0.25)
    Q3 = df_vitalsign['heartrate'].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df_vitalsign['heartrate'] = df_vitalsign['heartrate'].apply(lambda x: x if lower_bound <= x <= upper_bound else np.nan)

    numeric_cols = [ 'heartrate']

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_vitalsign[numeric_cols])
    plt.title("Box Plot VitalSign (HEARTRATE)")
    plt.xticks(rotation=45)
    plt.show()


            
    # Καθαρίζω τη στήλη pain του DataFrame df_triage, διαχειριζόμενος τις ελλείπουσες τιμές.
    # Όταν υπάρχουν ελλείπουσες (NaN) τιμές στη στήλη, αυτές αντικαθίστανται με το string 'Unknown'.
    # Αυτό διασφαλίζει ότι δεν θα έχουμε κενές τιμές στη στήλη.
    df_triage['pain'] = df_triage['pain'].fillna('Unknown').apply(clean_pain)

    df_triage['pain'].unique()

    # Define aggregation rules for all columns
    agg_rules = {
        'chiefcomplaint': lambda x: ', '.join(x.dropna().unique()),  # Concatenate symptoms
        'temperature': 'mean',  # Numerical aggregation
        'heartrate': 'mean',
        'resprate': 'mean',
        'o2sat': 'mean',
        'sbp': 'mean',
        'dbp': 'mean',
        'pain': lambda x: x.dropna().apply(str).str.cat(sep=', '),  # Concatenate pain values as strings
        'acuity': 'mean'
    }

    # Ομαδοποιεί το DataFrame df_triage με βάση το subject_id και συνοψίζει τα δεδομένα χρησιμοποιώντας προκαθορισμένους κανόνες για κάθε στήλη. 
    # Με την εντολή agg(agg_rules) εφαρμόζονται οι κανόνες που ορίστηκαν παραπάνω για κάθε ομάδα.
    # Το αποτέλεσμα είναι ένα νέο DataFrame, όπου κάθε subject_id εμφανίζεται μία φορά, συνοδευόμενο από συνοπτικές πληροφορίες που προκύπτουν 
    # από τους κανόνες αυτούς. Η χρήση της reset_index() διασφαλίζει ότι το subject_id μετατρέπεται ξανά σε κανονική στήλη (και όχι σε index), 
    # δημιουργώντας ένα νέο, συνοπτικό DataFrame με το όνομα df_triage_agg.
    df_triage_agg = df_triage.groupby('subject_id').agg(agg_rules).reset_index()

    # Αν υπάρχει ήδη άλλη στήλη με παρόμοιο όνομα σε άλλο DataFrame, η μετονομασία βοηθά να αποφεύγεται η σύγχυση όταν ενώνονται διαφορετικά DataFrames. 
    # Για παράδειγμα, temperature_triage υποδεικνύει ότι η θερμοκρασία αυτή αφορά το triage.
    # Προσθέτοντας το επίθεμα _triage, οι στήλες γίνονται πιο περιγραφικές. 
    df_triage_agg = df_triage_agg.rename(columns={
        'subject_id': 'subject_id', 'chiefcomplaint': 'chiefcomplaint', 'temperature': 'temperature_triage', 'heartrate' : 'heartrate_triage', 'resprate':'resprate_triage',
           'o2sat':'o2sat_triage', 'sbp':'sbp_triage', 'dbp':'dbp_triage', 'pain':'pain_triage', 'acuity':'acuity_triage'
    })



    df_triage_agg['pain_triage'] = convert_string_list(df_triage_agg['pain_triage'])

    # Μετατροπή του charttime σε τύπο datetime:
    # Αυτό διασφαλίζει ότι οι ημερομηνίες και ώρες της στήλης charttime αντιμετωπίζονται σωστά ως χρονικές πληροφορίες, 
    # επιτρέποντας την εκτέλεση χρονικών υπολογισμών. 
    # Με αυτόν τον τρόπο, μπορείς να υπολογίσεις την πρώτη (ελάχιστη), τελευταία (μέγιστη) χρονική εγγραφή καθώς και το συνολικό χρονικό εύρος για κάθε ασθενή.
    df_vitalsign['charttime'] = pd.to_datetime(df_vitalsign['charttime'])

    # Οι κανόνες aggregation καθορίζουν πώς να συνοψιστούν οι τιμές για κάθε στήλη. 
    # Για τη charttime, υπολογίζονται:
    # Η πρώτη χρονική εγγραφή (min).
    # Η τελευταία χρονική εγγραφή (max).
    # Η χρονική διαφορά μεταξύ πρώτης και τελευταίας καταγραφής (σε δευτερόλεπτα).
    # Για αριθμητικές στήλες όπως temperature, heartrate, resprate, υπολογίζεται ο μέσος όρος.
    # Για μη αριθμητικές στήλες, όπως rhythm, οι μοναδικές μη κενές τιμές συνενώνονται σε μία ενιαία συμβολοσειρά, διαχωρισμένες με κόμμα
    # Αυτό παρέχει μια συνοπτική εικόνα όλων των διακριτών τιμών που καταγράφηκαν για τον ασθενή.
    # Όλα τα δεδομένα ομαδοποιούνται ανά subject_id. Με αυτό τον τρόπο, όλες οι εγγραφές που ανήκουν στον ίδιο ασθενή συνοψίζονται σε μία γραμμή.
      
    # Define aggregation rules
    vitalsign_agg_rules = {
        'charttime': ['min', 'max', lambda x: (x.max() - x.min()).total_seconds()],  # First, last, and time range
        'temperature': 'mean',
        'heartrate': 'mean',
        'resprate': 'mean',
        'o2sat': 'mean',
        'sbp': 'mean',
        'dbp': 'mean',
        'pain': lambda x: f"Mean: {x.mean():.2f}" if pd.api.types.is_numeric_dtype(x) else ', '.join(x.dropna().unique()),
        'rhythm': lambda x: ', '.join(x.dropna().unique())
    }

    # Το αποτέλεσμα είναι ένα νέο DataFrame df_vitalsign_agg, όπου κάθε subject_id αντιστοιχεί σε μια μοναδική γραμμή.
    # Κάθε subject_id εμφανίζεται τώρα μόνο μία φορά. Όλες οι σχετικές μετρήσεις του έχουν συγκεντρωθεί 
    # και συνοψιστεί στις αντίστοιχες στήλες. Το DataFrame περιέχει πλέον πολύ λιγότερες σειρές, αλλά περισσότερη πληροφορία σε κάθε γραμμή.
    df_vitalsign_agg = df_vitalsign.groupby('subject_id').agg(vitalsign_agg_rules).reset_index()

    # Create a mapping from the MultiIndex tuples to the desired column names
    new_column_names = {
        ('subject_id', ''): 'subject_id',
        ('charttime', 'min'): 'charttime_min',
        ('charttime', 'max'): 'charttime_max',
        ('charttime', '<lambda_0>'): 'charttime_lambda',
        ('temperature', 'mean'): 'temperature_mean_vitalsign',
        ('heartrate', 'mean'): 'heartrate_mean_vitalsign',
        ('resprate', 'mean'): 'resprate_mean_vitalsign',
        ('o2sat', 'mean'): 'o2sat_mean_vitalsign',
        ('sbp', 'mean'): 'sbp_mean_vitalsign',
        ('dbp', 'mean'): 'dbp_mean_vitalsign',
        ('pain', '<lambda>'): 'pain_vitalsign',
        ('rhythm', '<lambda>'): 'rhythm_vitalsign',
    }

    # Rename the columns
    df_vitalsign_agg.columns = [new_column_names[col] for col in df_vitalsign_agg.columns]

    df_vitalsign_agg['pain_vitalsign'] = convert_string_list(df_vitalsign_agg['pain_vitalsign'] )
    df_vitalsign_agg['rhythm_vitalsign'] = convert_string_list(df_vitalsign_agg['rhythm_vitalsign'] )

    df_medrecon['charttime'] = pd.to_datetime(df_medrecon['charttime'], errors='coerce')

    # Define aggregation rules
    agg_rules = {
        'stay_id': lambda x: ', '.join(map(str, x.unique())),  # Συνδυάζει όλα τα μοναδικά stay_id του ασθενή σε ένα πεδίο.
        'charttime': ['min', 'max'],  # Υπολογίζει τον πρώτο (min) και τελευταίο (max) χρόνο ανασκόπησης φαρμάκων.
        'name': lambda x: ', '.join(x.dropna().unique()),  # Συνδυάζει όλα τα μοναδικά ονόματα φαρμάκων που ανασκοπήθηκαν.
        'gsn': lambda x: ', '.join(map(str, x.unique())),  # Συνδυάζει όλους τους μοναδικούς κωδικούς που σχετίζονται με τα φάρμακα.
        'ndc': lambda x: ', '.join(map(str, x.unique())),  
        'etc_rn': lambda x: ', '.join(map(str, x.unique())),  
        'etccode': lambda x: ', '.join(map(str, x.unique())),  
        'etcdescription': lambda x: ', '.join(x.dropna().unique())  # Συνδυάζει όλες τις μοναδικές κατηγορίες φαρμάκων που χρησιμοποιήθηκαν.
    }

    # Ομαδοποίηση με βάση τον subject_id, δηλαδή για κάθε ασθενή ξεχωριστά.
    # Χρησιμοποιεί τους κανόνες aggregation (agg_rules) για να δημιουργήσει μια συνοπτική εικόνα ανά ασθενή.
    df_medrecon_agg = df_medrecon.groupby('subject_id').agg(agg_rules).reset_index()

    # Ομαλοποίηση των Ονομάτων Στηλών
    # Μετά την .agg() μέθοδο, οι στήλες charttime (min & max) έχουν multi-index columns, δηλαδή ονόματα στηλών με δύο επίπεδα.
    # Αυτό σημαίνει ότι αντί να έχουμε μια απλή στήλη charttime_min, η τελική στήλη αποθηκεύεται με ένα ιεραρχικό όνομα,
    # η στήλη charttime περιέχει από κάτω τις υπο-στήλες min και max
    # Μετατρέπει αυτά τα multi-level ονόματα σε απλά strings στη μορφή charttime_min, charttime_max, name, etcdescription κάνοντας τα πιο ευανάγνωστα.
    df_medrecon_agg.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in df_medrecon_agg.columns]

    # Μετονομασία Στηλών
    df_medrecon_agg.rename(columns={
        'charttime_min': 'first_reconciliation_time',
        'charttime_max': 'last_reconciliation_time',
        'name': 'medications_reconciled',
        'etcdescription': 'medication_categories'
    }, inplace=True)

    df_medrecon_agg = df_medrecon_agg.rename(columns={
        'subject_id':'subject_id', 'stay_id_<lambda>': 'stay_id', 'first_reconciliation_time': 'first_reconciliation_time',
           'last_reconciliation_time': 'last_reconciliation_time', 'name_<lambda>' : 'name_med', 'gsn_<lambda>': 'gsn_med',
           'ndc_<lambda>': 'ndc_med', 'etc_rn_<lambda>': 'etc_rn_med', 'etccode_<lambda>': 'etccode_med',
           'etcdescription_<lambda>': 'etcdescription'
    })

    df_medrecon_agg['etcdescription'] = convert_string_list(df_medrecon_agg['etcdescription'] )
    df_medrecon_agg['etccode_med'] = convert_string_list(df_medrecon_agg['etccode_med'] )
    df_medrecon_agg['etc_rn_med'] = convert_string_list(df_medrecon_agg['etc_rn_med'] )
    df_medrecon_agg['ndc_med'] = convert_string_list(df_medrecon_agg['ndc_med'] )
    df_medrecon_agg['gsn_med'] = convert_string_list(df_medrecon_agg['gsn_med'] )
    df_medrecon_agg['name_med'] = convert_string_list(df_medrecon_agg['name_med'] )
    df_medrecon_agg['stay_id'] = convert_string_list(df_medrecon_agg['stay_id'] )

    # Μετατροπή της charttime σε μορφή datetime
    # Αν κάποια τιμή δεν μπορεί να μετατραπεί (π.χ. είναι κενή ή μη έγκυρη), τότε:
    # Με την επιλογή errors='coerce', η μη έγκυρη τιμή μετατρέπεται σε NaT (Not a Time) αντί να εμφανιστεί σφάλμα.
    df_pyxis['charttime'] = pd.to_datetime(df_pyxis['charttime'], errors='coerce')

    # Διαχείριση των ελλειπουσών τιμών στη στήλη gsn
    # Ελέγχω τη στήλη gsn, που περιέχει GSN (Generic Sequence Numbers), έναν μοναδικό αναγνωριστικό αριθμό για φάρμακα.
    # Εφόσον η στήλη gsn περιέχει αριθμούς, η μετατροπή της σε int βελτιώνει τη διαχείρισή της και μειώνει την κατανάλωση μνήμης.
    # Αντικαθιστάω όλες τις NaN τιμές με -1. Η τιμή -1 χρησιμοποιείται ως placeholder για να δείξει "άγνωστη τιμή".
    # Τέλος, μετατρέπω τη στήλη σε ακέραιο αριθμό (int).
    df_pyxis['gsn'] = df_pyxis['gsn'].fillna(-1).astype(int)  # Replace NaN with -1 (or another placeholder)

    # Define aggregation rules
    agg_rules = {
        'stay_id': lambda x: ', '.join(map(str, x.unique())),  # Συνενώνει όλα τα μοναδικά stay_id του ασθενή σε ένα string
        'charttime': ['min', 'max'],  # Υπολογίζει τον πρώτο (min) και τελευταίο (max) χρόνο χορήγησης φαρμάκων.
        'name': lambda x: ', '.join(x.unique()),  # Συνενώνει όλα τα μοναδικά ονόματα φαρμάκων που δόθηκαν στον ασθενή
        'gsn_rn': lambda x: ', '.join(map(str, x.unique())),  # Συνενώνει όλους τους μοναδικούς GSN-RN κωδικούς των φαρμάκων.
        'gsn': lambda x: ', '.join(map(str, x.unique()))  # Συνενώνει όλους τους μοναδικούς GSN κωδικούς των φαρμάκων.
    }

    # Ομαδοποίηση Δεδομένων με Βάση το subject_id δηλαδή, για κάθε ασθενή.
    # Εφαρμόζω τους agg_rules που όρισα προηγουμένως.
    # Επαναφέρω το index (reset_index()), ώστε το subject_id να γίνει ξανά κανονική στήλη.
    # Ως αποτέλεσμα δημιουργείται ένα νέο DataFrame df_pyxis_agg, όπου κάθε γραμμή αντιστοιχεί σε έναν ασθενή, 
    # με συνοπτικά στοιχεία για τα φάρμακα που του χορηγήθηκαν.
    df_pyxis_agg = df_pyxis.groupby('subject_id').agg(agg_rules).reset_index()

    # Ομαλοποίηση των Στηλών (Flatten Multi-Level Columns)
    df_pyxis_agg.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in df_pyxis_agg.columns]

    # Μετονομασία Στηλών για να είναι πιο κατανοητές
    df_pyxis_agg.rename(columns={
        'charttime_min': 'first_administration_time', # Πρώτη χορήγηση φαρμάκου.
        'charttime_max': 'last_administration_time', # Τελευταία χορήγηση φαρμάκου.
        'name': 'medications_administered', # Φάρμακα που χορηγήθηκαν.
        'gsn': 'gsn_codes', # Κωδικοί GSN φαρμάκων.
        'gsn_rn': 'gsn_rn_codes', #  Κωδικοί GSN-RN φαρμάκων.
        'stay_id': 'stay_ids' # Όλα τα stay_id του ασθενή.
    }, inplace=True)

    df_pyxis_agg = df_pyxis_agg.rename(columns={
        'subject_id': 'subject_id', 'stay_id_<lambda>': 'stay_id', 'first_administration_time':'first_administration_time',
           'last_administration_time':'last_administration_time', 'name_<lambda>': 'name_py', 'gsn_rn_<lambda>':'gsn_rn_py',
           'gsn_<lambda>': 'gsn_py'
    })

    # Μεταρέπει τιμές μιας στήλης από μορφή κειμένου (string) σε λίστα (list).
    # Στην προηγούμενη διαδικασία ομαδοποίησης (aggregation), οι τιμές σε αυτές τις στήλες συνενώθηκαν με ', '.join(...) σε strings.
    # Επειδή το pandas θεωρεί ότι η στήλη είναι απλό κείμενο (string), δεν μπορείς εύκολα να αναζητήσεις στοιχεία μέσα της.
    # Τώρα η στήλη είναι πραγματική λίστα, μπορεί να εφαρμόστεί φιλτράρισμα
    df_pyxis_agg['gsn_py'] = convert_string_list(df_pyxis_agg['gsn_py'] )
    df_pyxis_agg['gsn_rn_py'] = convert_string_list(df_pyxis_agg['gsn_rn_py'] )
    df_pyxis_agg['name_py'] = convert_string_list(df_pyxis_agg['name_py'] )
    df_pyxis_agg['stay_id'] = convert_string_list(df_pyxis_agg['stay_id'] )

    # Μετατροπή των intime και outtime σε datetime
    df_edstays['intime'] = pd.to_datetime(df_edstays['intime'], errors='coerce')
    df_edstays['outtime'] = pd.to_datetime(df_edstays['outtime'], errors='coerce')

    # Συμπλήρωση ελλειπουσών τιμών στη στήλη hadm_id
    # Το (Hospital Admission ID είναι ένας μοναδικός αριθμός εισαγωγής στο νοσοκομείο. Αν λείπει, το σύστημα δεν μπορεί να το συσχετίσει με συγκεκριμένη εισαγωγή.
    # Εντοπίζει αν υπάρχουν NaN (κενές τιμές) στη στήλη hadm_id.
    # Αντικαθιστά αυτές τις κενές τιμές με -1 ή άλλο προκαθορισμένο placeholder.
    # Μετατρέπει ολόκληρη τη στήλη σε ακέραιο αριθμό (int), ώστε να είναι συνεπής.
    df_edstays['hadm_id'] = df_edstays['hadm_id'].fillna(-1).astype(int)

    # Define aggregation rules
    agg_rules = {
        'hadm_id': lambda x: ', '.join(map(str, x.unique())),  # Συνένωση μοναδικών ID νοσηλείας σε ένα string.
        'stay_id': lambda x: ', '.join(map(str, x.unique())),  # Συνένωση μοναδικών ID παραμονής σε ένα string.
        'intime': 'min',  # Πρώτη χρονική στιγμή εισαγωγής
        'outtime': 'max',  # Τελευταία χρονική στιγμή εξόδου
        'gender': 'first',  # Το πρώτο φύλο (υποθέτοντας ότι είναι σταθερό)
        'race': 'first',  
        'arrival_transport': lambda x: ', '.join(x.dropna().unique()),  # Συνενώνει όλους τους μοναδικούς τρόπους μεταφοράς (άφιξης) του ασθενή στο νοσοκομείο.
        'disposition': lambda x: ', '.join(x.dropna().unique())  # Συνένωση μοναδικών καταστάσεων εξόδου
    }

    # Ομαδοποίηση Δεδομένων με Βάση το subject_id
    # Δημιουργείται ένα νέο DataFrame df_edstays_agg, όπου κάθε γραμμή αντιστοιχεί σε έναν ασθενή, με όλες τις σχετικές πληροφορίες του συνοπτικά αποθηκευμένες.
    df_edstays_agg = df_edstays.groupby('subject_id').agg(agg_rules).reset_index()

    # Μετονομασία Στηλών
    df_edstays_agg.rename(columns={
        'intime': 'earliest_admission_time', # Πρώτη χρονική στιγμή εισαγωγής.
        'outtime': 'latest_discharge_time', # Τελευταία χρονική στιγμή εξόδου.
        'stay_id': 'stay_ids', # Όλες οι παραμονές του ασθενή.
        'hadm_id': 'hospital_admission_ids', # Όλες οι νοσηλείες του ασθενή.
        'arrival_transport': 'arrival_modes', # Τρόποι άφιξης στο νοσοκομείο.
        'disposition': 'dispositions' # Καταστάσεις εξόδου από το νοσοκομείο.
    }, inplace=True)

    df_edstays_agg['arrival_modes'] = convert_string_list(df_edstays_agg['arrival_modes'] )
    df_edstays_agg['dispositions'] = convert_string_list(df_edstays_agg['dispositions'] )
    df_edstays_agg['stay_ids'] = convert_string_list(df_edstays_agg['stay_ids'] )
    df_edstays_agg['hospital_admission_ids'] = convert_string_list(df_edstays_agg['hospital_admission_ids'] )

    # Define aggregation rules
    agg_rules = {
        'stay_id': lambda x: ', '.join(map(str, x.unique())),  # Συνενώνει όλα τα μοναδικά stay_id που αντιστοιχούν στον ασθενή.
        'seq_num': lambda x: ', '.join(map(str, x.unique())),  # Συνενώνει όλους τους μοναδικούς αριθμούς σειράς διάγνωσης (seq_num) που χρησιμοποιούνται για την κατάταξη των διαγνώσεων του ασθενή.
        'icd_code': lambda x: ', '.join(x.unique()),  # Συνενώνει όλους τους μοναδικούς κωδικούς ICD που αντιστοιχούν στις διαγνώσεις του ασθενή.
        'icd_version': 'first',  # Κρατάει την πρώτη καταγραφή της έκδοσης ICD που χρησιμοποιήθηκε. (Συνήθως οι ICD εκδόσεις δεν αλλάζουν για έναν ασθενή).
        'icd_title': lambda x: ', '.join(x.unique())  # Συνενώνει όλους τους μοναδικούς τίτλους ICD των διαγνώσεων του ασθενή.
    }

    # Ομαδοποίηση Δεδομένων με Βάση το subject_id
    # Δημιουργείται ένα νέο DataFrame df_diagnosis_agg, όπου κάθε γραμμή αντιστοιχεί σε έναν ασθενή, 
    # με όλες τις σχετικές διαγνώσεις του συνοπτικά αποθηκευμένες.
    df_diagnosis_agg = df_diagnosis.groupby('subject_id').agg(agg_rules).reset_index()

    # Μετονομασία Στηλών
    df_diagnosis_agg.rename(columns={
        'stay_id': 'stay_ids', # Όλες οι παραμονές του ασθενή στο νοσοκομείο.
        'seq_num': 'sequence_numbers', # Οι αριθμοί σειράς των διαγνώσεων.
        'icd_code': 'icd_codes', # Όλοι οι ICD κωδικοί των διαγνώσεων.
        'icd_title': 'diagnoses' # Όλοι οι τίτλοι των διαγνώσεων.
    }, inplace=True)

    df_diagnosis_agg['diagnoses'] = convert_string_list(df_diagnosis_agg['diagnoses'] )
    df_diagnosis_agg['icd_codes'] = convert_string_list(df_diagnosis_agg['icd_codes'] )
    df_diagnosis_agg['sequence_numbers'] = convert_string_list(df_diagnosis_agg['sequence_numbers'] )
    df_diagnosis_agg['stay_ids'] = convert_string_list(df_diagnosis_agg['stay_ids'] )

    # Τα συνοπτικά DataFrames (diagnosis, edstays, triage και vitalsign) συγχωνεύονται με pd.merge 
    # με εξωτερική ένωση (outer join) βασισμένα στη στήλη subject_id ώστε να διατηρηθούν όλες οι εγγραφές για κάθε ασθενή.
    # Δεν χάνεις πληροφορίες από κανένα DataFrame, ακόμη και αν κάποιο subject_id υπάρχει μόνο σε ένα από αυτά.
    # Ο συνδυασμός αυτών των DataFrames επιτρέπει να έχεις όλες τις πληροφορίες για έναν ασθενή (subject_id) σε ένα ενιαίο DataFrame.
    merged_df_diagnosis = pd.merge(df_diagnosis_agg, df_edstays_agg, on="subject_id", how="outer")
    print(merged_df_diagnosis.shape)

    merged_df_diagnosis = pd.merge(merged_df_diagnosis,df_triage_agg , on="subject_id", how="outer")
    print(merged_df_diagnosis.shape)

    merged_df_diagnosis = pd.merge(merged_df_diagnosis, df_vitalsign_agg, on="subject_id", how="outer")
    print(merged_df_diagnosis.shape)

    # Παρόμοια, δημιουργείται συγχωνευμένο DataFrame που περιέχει πληροφορίες για τη φαρμακευτική αγωγή (medrecon, pyxis)
    merged_df_treatment = pd.merge(df_medrecon_agg, df_pyxis_agg, on="subject_id", how="outer")
    print(merged_df_treatment.shape)


    # Apply the function to create a new column
    merged_df_diagnosis['combined_text'] = merged_df_diagnosis.apply(combine_fields, axis=1)


    # Apply the function to create a new column
    merged_df_treatment['combined_text_treatment'] = merged_df_treatment.apply(combine_fields_treatment, axis=1)

    # Generate embeddings for the entire column
    texts = merged_df_diagnosis['combined_text'].tolist()
    embeddings = generate_embeddings_batch(texts)
    merged_df_diagnosis['embedding'] = list(embeddings)

    # Save df_subset to a pickle file
    merged_df_diagnosis.to_pickle("merged_df_diagnosis.pkl")
    print("DataFrame saved as merged_df_diagnosis.pkl")

	    
    # Extract embeddings and convert to numpy array
    embeddings = np.stack(merged_df_diagnosis['embedding'].values).astype('float32')

    # Normalize embeddings
    faiss.normalize_L2(embeddings)  # <==== ADD THIS!

    # Ensure no NaN values in embeddings
    if np.isnan(embeddings).any():
        raise ValueError("Embeddings contain NaN values. Check data preprocessing.")

    d = 1024
    if embeddings.shape[1] != d:
        raise ValueError(f"Expected embedding dimension {d}, but got {embeddings.shape[1]}")
    index = faiss.IndexFlatL2(d)


    # Add embeddings to the index
    index.add(embeddings)
    faiss.write_index(index, "faiss_index_merged_df_diagnosis.idx")
    print("FAISS index saved to faiss_index_merged_df_diagnosis.idx")


    # Generate embeddings for the entire column
    texts = merged_df_treatment['combined_text_treatment'].tolist()
    embeddings = generate_embeddings_batch(texts)
    merged_df_treatment['embedding'] = list(embeddings)

    # Save df_subset to a pickle file
    merged_df_treatment.to_pickle("merged_df_treatment.pkl")
    print("DataFrame saved as merged_df_treatment.pkl")

    # Extract embeddings and convert to numpy array
    embeddings = np.stack(merged_df_treatment['embedding'].values).astype('float32')

    # Normalize embeddings
    faiss.normalize_L2(embeddings)  # <==== ADD THIS!

    if np.isnan(embeddings).any():
        raise ValueError("Embeddings contain NaN values. Check data preprocessing.")

    d = 1024
    if embeddings.shape[1] != d:
        raise ValueError(f"Expected embedding dimension {d}, but got {embeddings.shape[1]}")
    index = faiss.IndexFlatL2(d)

    # Add embeddings to the index
    index.add(embeddings)

    faiss.write_index(index, "faiss_index_merged_df_treatment.idx")
    print("FAISS index saved to faiss_index_merged_df_treatment.idx")



if __name__ == '__main__':
    main()