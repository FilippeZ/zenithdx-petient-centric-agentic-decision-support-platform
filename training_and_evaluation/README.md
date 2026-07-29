# ZenithDx Training, Fine-Tuning & Evaluation Suite

This folder consolidates all scripts, notebooks, dataset preprocessors, training pipelines, and evaluation benchmarks used to train and evaluate the AI models underpinning the **ZenithDx Patient-Centric Agentic Clinical Decision Support Platform**.

---

## 📁 Repository Structure

```
training_and_evaluation/
├── README.md
├── llm_doctor2/
│   ├── fine_tuning/
│   │   ├── alpaca_doctor.py       # Converts clinical Q&A & case studies into Alpaca instruction format
│   │   ├── alpaca_pharm.py        # Converts pharmaceutical dosage & interaction data into Alpaca format
│   │   ├── train_doctor.py        # Unsloth + LoRA 4-bit fine-tuning pipeline for Llama 3.2 3B
│   │   └── train_pharm.py         # Secondary fine-tuning pass for pharmacology reasoning
│   ├── evaluation/
│   │   ├── generate_predictions_doctor.py  # Generates test set predictions using fine-tuned weights
│   │   ├── evaluate_predictions_doctor.py  # Computes clinical accuracy, BLEU, ROUGE, and BERTScore
│   │   ├── evaluate_predictions_pharm.py   # Computes drug interaction prediction precision/recall
│   │   └── doctor2_predictions.json        # Test set predictions benchmark output
│   └── gguf_export/
│       ├── convert_hf_to_gguf.py  # Converts HuggingFace LoRA merged model to GGUF format
│       ├── convert_lora_to_gguf.py# Exports GGUF adapter tensors for local Ollama deployment
│       └── Modelfile              # Ollama model definition template with system prompts & params
│
├── vision_models/
│   ├── resnet50_classification/
│   │   ├── MultiLabel_Image_Classification_.ipynb # Jupyter notebook for ResNet-50 training & loss curves
│   │   ├── multilabel_image_classification_.py    # Standalone PyTorch ResNet-50 multi-label training script
│   │   ├── preprocess_csv.py                      # Preprocesses ChestX-ray14 CSV labels and split indices
│   │   ├── dataset.py                             # PyTorch Dataset loader with data augmentation
│   │   ├── model.py                               # ResNet-50 architecture definition with custom FC head
│   │   ├── train.py                               # Training loop with BCEWithLogitsLoss and threshold optimization
│   │   ├── evaluate.py                            # Computes per-pathology ROC-AUC and optimal thresholds
│   │   └── class_eval.py                          # Multi-label classification metric evaluation
│   └── sa_unet_segmentation/
│       ├── train_seg.py                           # Keras/TensorFlow S²A-UNet lung segmentation trainer
│       └── seg_eval.py                            # Dice Similarity Coefficient (DSC) & IoU metric calculator
│
└── graph_hgt_ehr/
    ├── preprocessing/
    │   ├── data_preprocessing.py # MIMIC-IV-ED cleaning, vital sign normalization, and triage encoding
    │   ├── faiss_index_D.py      # SciBERT feature extraction and FAISS dense vector indexing
    │   └── faissQ.py             # Query processing for dense longitudinal trajectory search
    ├── graph_construction/
    │   └── hgt_model.py          # PyTorch Geometric Heterogeneous Graph Transformer (HGT) architecture
    └── hgt_training_eval/
        └── clustering.py         # K-Means clustering on HGT node embeddings for patient trajectory matching
```

---

## 🧠 1. Fine-Tuned Medical LLM (Doctor2)

### 1.1 Methodology
- **Base Backbone**: `Llama 3.2 3B Instruct` / `Llama 3.1 8B Instruct`
- **Fine-Tuning Technique**: Unsloth 4-bit Quantization + QLoRA (Rank $r=16$, $\alpha=16$, target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).
- **Instruction Dataset**: Prepared via `alpaca_doctor.py` using clinical ED case studies, MIMIC-IV discharge summaries, and evidence-based diagnostic protocols formatted as instruction-input-output triplets.
- **Quantization & Deployment**: Exported to GGUF (`convert_hf_to_gguf.py`) and packaged for low-latency local inference in Ollama via `Modelfile`.

### 1.2 Training & Evaluation Execution
```bash
# Prepare Alpaca format dataset
python llm_doctor2/fine_tuning/alpaca_doctor.py

# Fine-tune Llama 3.2 3B with Unsloth QLoRA
python llm_doctor2/fine_tuning/train_doctor.py

# Evaluate test set predictions
python llm_doctor2/evaluation/evaluate_predictions_doctor.py
```

---

## 🫁 2. Vision Models (S²A-UNet & ResNet-50)

### 2.1 Spatial Attention UNet (S²A-UNet)
- **Objective**: Automatic pulmonary boundary segmentation on raw chest X-rays to isolate lung parenchyma prior to pathology classification.
- **Metrics**: Achieves **94.8% Dice Similarity Coefficient (DSC)** and **90.1% Intersection over Union (IoU)**.

### 2.2 ResNet-50 Pathology Classifier
- **Objective**: Multi-label detection across 14 chest pathologies (Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule, Pneumonia, Pneumothorax, Consolidation, Edema, Emphysema, Fibrosis, Pleural Thickening, Hernia).
- **Explainability**: Integrated with Grad-CAM (`layer4`) to project spatial visual heatmaps onto segmented lung fields.

### 2.3 Training & Evaluation Execution
```bash
# Preprocess dataset splits
python vision_models/resnet50_classification/preprocess_csv.py

# Train PyTorch ResNet-50 classifier
python vision_models/resnet50_classification/train.py

# Evaluate per-pathology ROC-AUC and optimal decision thresholds
python vision_models/resnet50_classification/evaluate.py
```

---

## 🕸️ 3. Heterogeneous Graph Transformer (HGT) & Longitudinal EHR

### 3.1 Architecture & Pipeline
- **Dataset**: MIMIC-IV-ED (Emergency Department stays, vital signs, triage scores, diagnoses).
- **Graph Nodes**: `Patient`, `Stay`, `Symptom`, `VitalSign`, `Diagnosis`.
- **Relational Edges**: `presents_with`, `exhibits_vitals`, `diagnosed_with`, `longitudinal_next`.
- **HGT Embeddings**: Multi-head target-specific attention mechanisms capturing complex multi-modal interactions.
- **Clustering**: K-Means on HGT embeddings combined with FAISS dense vector search to retrieve similar historical patient cohorts.

---

## 📊 Summary of Benchmark Results

| Model / Subsystem | Benchmark Metric | Score |
| :--- | :--- | :---: |
| **Doctor2 (Fine-Tuned Llama 3.2 3B)** | Clinical Diagnosis Accuracy | **96.28%** |
| **S²A-UNet** | Segmentation Dice Coefficient | **94.80%** |
| **ResNet-50** | Mean Multi-Label ROC-AUC | **0.864** |
| **HGT + K-Means Clustering** | Longitudinal Patient Retrieval Accuracy | **91.40%** |
| **Full Agent Pipeline** | Physician Task Completion (KLM) | **31.9s** |
| **Usability Benchmark** | Physician HSUS Score | **76.1 / 100** |
