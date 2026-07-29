import os
import pickle
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import faiss

# Ορισμός φακέλων
diseases_folder = "Diseases"
vitals_folder = "Vital_Signs"

# Συνάρτηση φόρτωσης TXT αρχείων από φάκελο
def load_txt_files(folder_path):
    texts = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                texts.append(f.read())
    return texts

# Φόρτωση δεδομένων από τους δύο φακέλους
diseases_texts = load_txt_files(diseases_folder)
vitals_texts = load_txt_files(vitals_folder)

# Συνδυασμός όλων των κειμένων σε μία λίστα
all_texts = diseases_texts + vitals_texts

# Δημιουργία Document objects για όλα τα κείμενα
docs = [Document(page_content=text) for text in all_texts]

# Αρχικοποίηση του μοντέλου embeddings (χρησιμοποιώντας ένα μοντέλο της HuggingFace)
embeddings_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    model_kwargs={"device": "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"}
)

# Εφαρμογή τμηματοποίησης με RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # Προσαρμόστε το μέγεθος του chunk αν χρειάζεται
    chunk_overlap=50     # Επικάλυψη για καλύτερη διατήρηση του context
)
chunks = text_splitter.split_documents(docs)
print(f"Συνολικά chunks: {len(chunks)}")

# Αποθήκευση των chunks σε αρχείο pickle (.pkl) για επαναχρησιμοποίηση
with open("chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)
print("Τα chunks αποθηκεύτηκαν στο αρχείο 'chunks.pkl'.")

# Δημιουργία ενιαίου FAISS index για όλα τα chunks
faiss_index = FAISS.from_documents(chunks, embeddings_model)
print("Ο ενιαίος FAISS index δημιουργήθηκε επιτυχώς!")

# Αποθήκευση του FAISS index σε αρχείο .idx
faiss.write_index(faiss_index.index, "faiss_index_D.idx")
print("Ο FAISS index αποθηκεύτηκε στο αρχείο 'faiss_index_D.idx'.")
