import numpy as np
import faiss
import pickle

# === SETTINGS ===
EMBED_PATH = "visit_patient_emb.npz"  # Το path που έχουν αποθηκευτεί τα embeddings
FAISS_INDEX_PATH = "faiss_patient_index.bin"
MAPPING_PATH = "faiss_patient_mapping.pkl"
USE_PATIENT_EMBED = False  # True: patient_emb, False: visit_emb

# === Φόρτωση embeddings και ids ===
data = np.load(EMBED_PATH)
if USE_PATIENT_EMBED:
    emb = data["patient_emb"]        # Shape: (num_patients, D)
    print("Loaded patient embeddings:", emb.shape)
    patient_ids = data["patient_ids"] # οι πραγματικοί subject_id
    ids = patient_ids                # Για κάθε index, το subject_id
else:
    emb = data["visit_emb"]          # Shape: (num_visits, D)
    print("Loaded visit embeddings:", emb.shape)
    visit2pat = data['visit2pat']    # visit2pat: array με real patient ids (subject_id) ανά visit
    ids = np.array(visit2pat)        # ids[i] = subject_id

# === Δημιουργία FAISS index ===
d = emb.shape[1]
index = faiss.IndexFlatL2(d)
index.add(emb)
print(f"Added {emb.shape[0]} embeddings to FAISS index.")

# === Mapping index → patient_id ===
index_map = dict(enumerate(ids))

# === Αποθήκευση FAISS index και mapping ===
faiss.write_index(index, FAISS_INDEX_PATH)
with open(MAPPING_PATH, "wb") as f:
    pickle.dump(index_map, f)
print(f"Saved FAISS index to {FAISS_INDEX_PATH} and mapping to {MAPPING_PATH}")

# --- ΣΥΝΑΡΤΗΣΗ: Αναζήτηση embeddings ΜΟΝΟ για τον συγκεκριμένο ασθενή ---
def search_faiss_for_patient(query_vec, patient_id, k=5):
    """
    Δέχεται query_vec (D,) ή (1, D), επιστρέφει τα k πιο κοντινά visit embeddings ΤΟΥ ΣΥΓΚΕΚΡΙΜΕΝΟΥ ΑΣΘΕΝΗ.
    """
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(MAPPING_PATH, "rb") as f:
        index_map = pickle.load(f)
    indices = [i for i, pid in index_map.items() if pid == patient_id]
    if len(indices) == 0:
        print(f"No embeddings found for patient_id={patient_id}")
        return [], []
    all_emb = index.reconstruct_n(0, index.ntotal)
    sub_emb = np.stack([all_emb[i] for i in indices])
    d = sub_emb.shape[1]
    sub_index = faiss.IndexFlatL2(d)
    sub_index.add(sub_emb)
    D, I = sub_index.search(query_vec.reshape(1, -1), k)
    neighbor_global_idx = [indices[i] for i in I[0]]
    return neighbor_global_idx, D[0]

if __name__ == "__main__":
    # Δείξε τα πρώτα 20 unique patient ids και min/max
    print("First 20 unique patient ids:", np.unique(ids)[:20])
    print("Min id:", np.min(ids), "Max id:", np.max(ids))

    # --- Δοκιμές σε διαφορετικά σενάρια ---
    for USE_PATIENT_EMBED in [False, True]:
        print("\n==== TEST: USE_PATIENT_EMBED =", USE_PATIENT_EMBED, "====")
        if USE_PATIENT_EMBED:
            emb = data["patient_emb"]
            ids = data["patient_ids"]
        else:
            emb = data["visit_emb"]
            ids = np.array(data["visit2pat"])

        # Βρες ασθενείς με πολλές και λίγες επισκέψεις
        unique_ids, counts = np.unique(ids, return_counts=True)
        patient_many = unique_ids[np.argmax(counts)]  # ασθενής με τις ΠΕΡΙΣΣΟΤΕΡΕΣ επισκέψεις
        patient_single = unique_ids[np.argmin(counts)]  # ασθενής με ΜΙΑ επίσκεψη

        for patient_id in [patient_many, patient_single]:
            print(f"\n-- Testing patient_id: {patient_id} (visits: {np.sum(ids == patient_id)}) --")
            patient_indices = np.where(ids == patient_id)[0]
            if len(patient_indices) == 0:
                print(f"Patient ID {patient_id} not found!")
                continue

            query_idx = patient_indices[0]
            query_vec = emb[query_idx]
            topk = 5
            neighbor_idx, distances = search_faiss_for_patient(query_vec, patient_id, k=topk)

            print(f"Top-{topk} closest embeddings for patient_id={patient_id}:")
            print("  [DEBUG] Distances:", distances)
            print("  [DEBUG] Neighbor Patient IDs:", [ids[i] for i in neighbor_idx])
            # Αναλυτικά
            for idx, dist in zip(neighbor_idx, distances):
                same_patient = ids[idx] == patient_id
                print(f"    Embedding idx: {idx}, Distance: {dist:.3f}, Patient: {ids[idx]}, Same patient? {same_patient}")

            # Έλεγχος: Η πρώτη απόσταση πρέπει να είναι 0 (αν το query ανήκει στον ασθενή)
            if distances[0] == 0:
                print("  [OK] Πρώτη απόσταση = 0 (ακριβές match).")
            else:
                print("  [WARNING] Πρώτη απόσταση ΔΕΝ είναι 0!")

            # Έλεγχος: Όλοι οι neighbors πρέπει να είναι ο ίδιος patient
            if all(ids[i] == patient_id for i in neighbor_idx):
                print("  [OK] Όλα τα neighbor embeddings ανήκουν στον ίδιο ασθενή.")
            else:
                print("  [ERROR] Βρέθηκε neighbor από άλλο ασθενή!")

    print("\n[INFO] Δοκιμάστηκε και για patient-level και για visit-level embeddings.")
