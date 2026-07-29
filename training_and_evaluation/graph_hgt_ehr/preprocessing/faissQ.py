from transformers import AutoTokenizer, AutoModel  
import torch
import numpy as np
import faiss
import pandas as pd
from sklearn.decomposition import PCA

def main():
    # Φόρτωση tokenizer και μοντέλου
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5")
    model = AutoModel.from_pretrained("BAAI/bge-large-en-v1.5")
    
    # Φόρτωση FAISS index
    index = faiss.read_index("faiss_index_merged_df_diagnosis.idx")
    
    # Ορισμός συσκευής (GPU αν είναι διαθέσιμη, αλλιώς CPU) και μεταφορά του μοντέλου στη συσκευή
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Ορισμός του κειμένου ερωτήματος
    query_text = "I had pneumonia and i cant breath so well"
    
    # Tokenization του κειμένου ερωτήματος και μεταφορά των tensors στη συσκευή
    inputs = tokenizer([query_text], return_tensors='pt', padding=True, truncation=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    # Υπολογισμός του αποτελέσματος του μοντέλου και εξαγωγή της ενσωμάτωσης του ερωτήματος μέσω mean pooling
    with torch.no_grad():
        outputs = model(**inputs)
    query_embedding = outputs.last_hidden_state.mean(dim=1).detach().cpu().numpy()
    
    print("Computed query embedding type:", type(query_embedding))
    print("Computed query embedding shape:", query_embedding.shape)
    print("Computed query embedding dtype:", query_embedding.dtype)
    
    # Αν το query_embedding έχει μορφή (1, 1, 1024), το σφίγγουμε σε (1, 1024)
    if query_embedding.ndim == 3 and query_embedding.shape[1] == 1:
        query_embedding = np.squeeze(query_embedding, axis=1)
        print("Squeezed query embedding shape:", query_embedding.shape)
    
    # Βεβαιώνουμε ότι η ενσωμάτωση έχει 2D μορφή και είναι τύπου float32
    if query_embedding.ndim != 2 or query_embedding.shape[0] != 1:
        query_embedding = query_embedding.reshape(1, -1)
    if query_embedding.dtype != np.float32:
        query_embedding = query_embedding.astype('float32')
    
    print("Final query embedding shape:", query_embedding.shape)
    print("Final query embedding dtype:", query_embedding.dtype)
    
    # Έλεγχος διάστασης του FAISS index
    print("FAISS index dimension:", index.d)
    
    # Εάν η διάσταση της ενσωμάτωσης του ερωτήματος δεν ταιριάζει με τη διάσταση του index,
    # εφαρμόζουμε PCA για μείωση από 1024 σε index.d (768) διαστάσεις.
    if query_embedding.shape[1] != index.d:
        print(f"Query embedding dimension ({query_embedding.shape[1]}) does not match FAISS index dimension ({index.d}).")
        # Φόρτωση του DataFrame για να χρησιμοποιήσουμε τις ενσωματώσεις του για εκπαίδευση του PCA.
        df = pd.read_pickle("merged_df_diagnosis.pkl")
        
        # Υποθέτουμε ότι το df["embedding"] περιέχει numpy arrays με διάσταση (1024,).
        # Στοίχιση τους για δημιουργία πίνακα με μορφή (n_samples, 1024).
        embeddings_sample = np.vstack(df["embedding"].values)
        print("Fitting PCA on sample embeddings with shape:", embeddings_sample.shape)
        
        # Εκπαίδευση PCA για μείωση της διάστασης σε index.d (768)
        pca = PCA(n_components=index.d)
        pca.fit(embeddings_sample)
        
        # Μετατροπή της ενσωμάτωσης του ερωτήματος χρησιμοποιώντας το εκπαιδευμένο PCA
        query_embedding = pca.transform(query_embedding)
        print("Adjusted query embedding shape after PCA:", query_embedding.shape)
    
    # Προαιρετικά: Αν υπάρχει λειτουργία μετατροπής GPU index σε CPU, χρησιμοποιήστε την. Αλλιώς υποθέτουμε ότι ο index είναι σε CPU.
    if hasattr(faiss, 'index_gpu_to_cpu'):
        index = faiss.index_gpu_to_cpu(index)
    else:
        print("faiss.index_gpu_to_cpu is not available; assuming the index is CPU-based.")
    
    # Εκτέλεση αναζήτησης στο FAISS index
    k = 5  # Αριθμός πλησιέστερων γειτόνων για ανάκτηση
    distances, indices = index.search(query_embedding, k)
    print("Nearest neighbor indices:", indices)
    print("Distances:", distances)
    
    # Φόρτωση του DataFrame που περιέχει τα κείμενα από το pickle αρχείο
    df = pd.read_pickle("merged_df_diagnosis.pkl")
    texts = df["combined_text"].tolist()
    
    # Ανάκτηση των κειμένων που αντιστοιχούν στους δείκτες των πλησιέστερων γειτόνων
    result_texts = [texts[i] for i in indices[0]]
    print("Retrieved texts:")
    for text in result_texts:
        print(text)

if __name__ == "__main__":
    main()
