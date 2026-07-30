# ZenithDx — Clinical Evaluation & XAI Execution Report
**Execution Timestamp:** 2026-07-30 12:08:11
**Suite Status:** 4/4 Usecases Executed Successfully (100% PASS)

--- 

## Use Case 1: Multi-Modal Acute Pneumonia Diagnostic Triage
*Patient presenting with severe dyspnea, high fever, and productive cough accompanied by a Chest X-ray scan.*

- **Patient Query / Symptoms:** `Severe shortness of breath, high fever 39°C, productive cough with rusty sputum, and sharp right-sided chest pain on inspiration.`
- **Uploaded Image Path:** `c:\Users\wwefi\OneDrive\Υπολογιστής\diploma-thesis\pneumonia.jpeg`
- **Patient ID (EHR):** `None`

### 📝 **Clinician Diagnosis Report**

### **Clinical Assessment**
- **Patient Presentation:** "### Imaging Findings: Atelectasis, Lung Lesion, Pneumonia, Edema, Consolidation"
- **Radiological Evaluation:** Atelectasis, Lung Lesion, Pneumonia, Edema, Consolidation.
- **Longitudinal EHR History:** None provided/retrieved.

---

### **Differential Diagnosis**
1. **Community-Acquired Pneumonia (CAP):** High clinical probability given acute respiratory symptoms, fever, and pulmonary presentation.
2. **Acute Asthma Exacerbation / Bronchospasm:** Secondary consideration for reactive lower airway obstruction.
3. **Acute Bronchitis:** Secondary consideration for acute inflammatory lower airway disease.
4. **Pleurisy / Pleural Inflammation:** Supported by respiratory discomfort and dyspnea on inspiration.

---

### **Final Diagnosis**
🎯 **Acute Lower Respiratory Infection / Pneumonia**

---

### **Diagnostic Rationale & Explanation**
The diagnostic rationale for **Acute Lower Respiratory Infection / Pneumonia** is derived by synthesizing the patient's presentation ("### Imaging Findings: Atelectasis, Lung Lesion, Pneumonia, Edema, Consolidation") with radiological findings from the uploaded chest radiograph and evidence-based consensus guidelines.

### 📊 **ResNet-50 Multi-Label Pathology Scores**

| Pathology | Probability | Assessment |
| :--- | :--- | :--- |
| **Atelectasis** | **61%** | 🟡 Moderate Risk |
| **Lung Lesion** | **60%** | 🟡 Moderate Risk |
| **Pneumonia** | **59%** | 🟡 Moderate Risk |
| **Edema** | **47%** | 🟡 Moderate Risk |
| **Consolidation** | **43%** | 🟡 Moderate Risk |

### 🫁 **Generated XAI Diagnostic Artifacts & Image Paths**

- 📷 **Original Chest X-ray:** `C:\Users\wwefi\OneDrive\Υπολογιστής\diploma-thesis\ZenithDx_Final\backend\outputs\default\pneumonia_original.png`
- 🔥 **Grad-CAM Overlay:** `C:\Users\wwefi\OneDrive\Υπολογιστής\diploma-thesis\ZenithDx_Final\backend\outputs\default\pneumonia_gradcam_overlay.png`
- 🫁 **Segmented Grad-CAM (S²A-UNet ROI):** `C:\Users\wwefi\OneDrive\Υπολογιστής\diploma-thesis\ZenithDx_Final\backend\outputs\default\pneumonia_gradcam_segmented.png`
- 📊 **Captum Text Attribution Plot:** `C:\Users\wwefi\OneDrive\Υπολογιστής\diploma-thesis\ZenithDx_Final\backend\outputs\default\20260730_120708\captum_query_20260730_120709_seq.png`
- 🧠 **captum_history_seq:** `None`
- 🧠 **captum_history_tok:** `None`
- 🧠 **captum_image_seq:** `C:\Users\wwefi\OneDrive\Υπολογιστής\diploma-thesis\ZenithDx_Final\backend\outputs\default\20260730_120708\captum_image_20260730_120710_seq.png`
- 🧠 **captum_image_tok:** `C:\Users\wwefi\OneDrive\Υπολογιστής\diploma-thesis\ZenithDx_Final\backend\outputs\default\20260730_120708\captum_image_20260730_120710_tok.png`
- 🧠 **captum_query_tok:** `C:\Users\wwefi\OneDrive\Υπολογιστής\diploma-thesis\ZenithDx_Final\backend\outputs\default\20260730_120708\captum_query_20260730_120709_tok.png`

#### **Top Attribution Tokens (PyTorch Captum)**

**Section: Query**
- `chest` (0.948), `shortness` (0.882), `pain` (0.796), `breath` (0.761), `fever` (0.743)

**Section: Image Findings**
- `lesion` (0.838), `pneumonia` (0.704), `atelectasis` (0.612), `edema` (0.529), `lung` (0.406)

--- 

## Use Case 2: Longitudinal EHR History Fusion & Respiratory Surveillance
*Patient with registered longitudinal EHR history (ID: 10000032) presenting with worsening dyspnea and bilateral congestion.*

- **Patient Query / Symptoms:** `Worsening shortness of breath over 4 days, bilateral lower lung congestion, persistent fatigue, and reduced exercise tolerance.`
- **Uploaded Image Path:** `c:\Users\wwefi\OneDrive\Υπολογιστής\diploma-thesis\pneumonia.jpeg`
- **Patient ID (EHR):** `10000032`

### 📝 **Clinician Diagnosis Report**

### **Clinical Assessment**
- **Patient Presentation:** "### Imaging Findings: Atelectasis, Lung Lesion, Pneumonia, Edema, Consolidation"
- **Radiological Evaluation:** Atelectasis, Lung Lesion, Pneumonia, Edema, Consolidation.
- **Longitudinal EHR History:** None provided/retrieved.

---

### **Differential Diagnosis**
1. **Community-Acquired Pneumonia (CAP):** High clinical probability given acute respiratory symptoms, fever, and pulmonary presentation.
2. **Acute Asthma Exacerbation / Bronchospasm:** Secondary consideration for reactive lower airway obstruction.
3. **Acute Bronchitis:** Secondary consideration for acute inflammatory lower airway disease.
4. **Pleurisy / Pleural Inflammation:** Supported by respiratory discomfort and dyspnea on inspiration.

---

### **Final Diagnosis**
🎯 **Acute Lower Respiratory Infection / Pneumonia**

---

### **Diagnostic Rationale & Explanation**
The diagnostic rationale for **Acute Lower Respiratory Infection / Pneumonia** is derived by synthesizing the patient's presentation ("### Imaging Findings: Atelectasis, Lung Lesion, Pneumonia, Edema, Consolidation") with radiological findings from the uploaded chest radiograph and evidence-based consensus guidelines.

### 📊 **ResNet-50 Multi-Label Pathology Scores**

| Pathology | Probability | Assessment |
| :--- | :--- | :--- |
| **Atelectasis** | **61%** | 🟡 Moderate Risk |
| **Lung Lesion** | **60%** | 🟡 Moderate Risk |
| **Pneumonia** | **59%** | 🟡 Moderate Risk |
| **Edema** | **47%** | 🟡 Moderate Risk |
| **Consolidation** | **43%** | 🟡 Moderate Risk |

### 🫁 **Generated XAI Diagnostic Artifacts & Image Paths**

- 📷 **Original Chest X-ray:** `C:\Users\wwefi\OneDrive\Υπολογιστής\diploma-thesis\ZenithDx_Final\backend\outputs\10000032\pneumonia_original.png`
- 🔥 **Grad-CAM Overlay:** `C:\Users\wwefi\OneDrive\Υπολογιστής\diploma-thesis\ZenithDx_Final\backend\outputs\10000032\pneumonia_gradcam_overlay.png`
- 🫁 **Segmented Grad-CAM (S²A-UNet ROI):** `C:\Users\wwefi\OneDrive\Υπολογιστής\diploma-thesis\ZenithDx_Final\backend\outputs\10000032\pneumonia_gradcam_segmented.png`
- 📊 **Captum Text Attribution Plot:** `C:\Users\wwefi\OneDrive\Υπολογιστής\diploma-thesis\ZenithDx_Final\backend\outputs\10000032\20260730_120737\captum_query_20260730_120738_seq.png`
- 🧠 **captum_history_seq:** `C:\Users\wwefi\OneDrive\Υπολογιστής\diploma-thesis\ZenithDx_Final\backend\outputs\10000032\20260730_120737\captum_history_20260730_120738_seq.png`
- 🧠 **captum_history_tok:** `C:\Users\wwefi\OneDrive\Υπολογιστής\diploma-thesis\ZenithDx_Final\backend\outputs\10000032\20260730_120737\captum_history_20260730_120738_tok.png`
- 🧠 **captum_image_seq:** `C:\Users\wwefi\OneDrive\Υπολογιστής\diploma-thesis\ZenithDx_Final\backend\outputs\10000032\20260730_120737\captum_image_20260730_120738_seq.png`
- 🧠 **captum_image_tok:** `C:\Users\wwefi\OneDrive\Υπολογιστής\diploma-thesis\ZenithDx_Final\backend\outputs\10000032\20260730_120737\captum_image_20260730_120738_tok.png`
- 🧠 **captum_query_tok:** `C:\Users\wwefi\OneDrive\Υπολογιστής\diploma-thesis\ZenithDx_Final\backend\outputs\10000032\20260730_120737\captum_query_20260730_120738_tok.png`

#### **Top Attribution Tokens (PyTorch Captum)**

**Section: Query**
- `breath` (0.785), `shortness` (0.719), `tolerance` (0.617), `persistent` (0.615), `reduced` (0.598)

**Section: Image Findings**
- `lesion` (0.838), `pneumonia` (0.704), `atelectasis` (0.612), `edema` (0.529), `lung` (0.406)

**Section: History**
- `pain` (0.895), `98.4` (0.645), `heartrate:` (0.645), `abdominal` (0.634), `resprate:` (0.621)

--- 

## Use Case 3: Text-Only Acute Febrile Clinical Consultation
*Pure text consultation without radiograph. Testing strict hallucination prevention (Image Path: None, History: None).*

- **Patient Query / Symptoms:** `High fever 38.8°C, sudden onset frontal headache, dry cough, sore throat, and generalized muscle aches for 2 days.`
- **Uploaded Image Path:** `None`
- **Patient ID (EHR):** `None`

### 📝 **Clinician Diagnosis Report**

### **Clinical Assessment**
- **Patient Presentation:** "### Imaging Findings: No chest X-ray was provided for this consultation."
- **Radiological Evaluation:** No chest X-ray was provided for this consultation.
- **Longitudinal EHR History:** None provided/retrieved.

---

### **Differential Diagnosis**
1. **Acute Viral Illness:** Primary differential for acute fever and systemic symptoms.
2. **Secondary Upper Airway Inflammation:** Differential candidate pending diagnostic workup.

---

### **Final Diagnosis**
🎯 **Acute Febrile Clinical Consultation (Under Evaluation)**

---

### **Diagnostic Rationale & Explanation**
The diagnostic rationale for **Acute Febrile Clinical Consultation (Under Evaluation)** is derived by synthesizing the patient's presentation ("### Imaging Findings: No chest X-ray was provided for this consultation.") with clinical symptom presentation and evidence-based consensus guidelines.

### 🫁 **Generated XAI Diagnostic Artifacts & Image Paths**

- 📊 **Captum Text Attribution Plot:** `C:\Users\wwefi\OneDrive\Υπολογιστής\diploma-thesis\ZenithDx_Final\backend\outputs\default\20260730_120751\captum_query_20260730_120752_seq.png`
- 🧠 **captum_history_seq:** `None`
- 🧠 **captum_history_tok:** `None`
- 🧠 **captum_image_seq:** `None`
- 🧠 **captum_image_tok:** `None`
- 🧠 **captum_query_tok:** `C:\Users\wwefi\OneDrive\Υπολογιστής\diploma-thesis\ZenithDx_Final\backend\outputs\default\20260730_120751\captum_query_20260730_120752_tok.png`

#### **Top Attribution Tokens (PyTorch Captum)**

**Section: Query**
- `headache` (0.917), `cough` (0.873), `fever` (0.700), `sudden` (0.605), `high` (0.585)

--- 

## Use Case 4: Multi-Modal Consolidation & Hypoxia Screening
*Patient presenting with hypoxia (SpO2 93%), pleuritic pain, and right basilar opacity on X-ray.*

- **Patient Query / Symptoms:** `Pleuritic chest pain on deep inspiration, localized right basilar dullness on percussion, mild hypoxia (SpO2 93%), and chills.`
- **Uploaded Image Path:** `c:\Users\wwefi\OneDrive\Υπολογιστής\diploma-thesis\pneumonia.jpeg`
- **Patient ID (EHR):** `None`

### 📝 **Clinician Diagnosis Report**

### **Clinical Assessment**
- **Patient Presentation:** "### Imaging Findings: Atelectasis, Lung Lesion, Pneumonia, Edema, Consolidation"
- **Radiological Evaluation:** Atelectasis, Lung Lesion, Pneumonia, Edema, Consolidation.
- **Longitudinal EHR History:** None provided/retrieved.

---

### **Differential Diagnosis**
1. **Community-Acquired Pneumonia (CAP):** High clinical probability given acute respiratory symptoms, fever, and pulmonary presentation.
2. **Acute Asthma Exacerbation / Bronchospasm:** Secondary consideration for reactive lower airway obstruction.
3. **Acute Bronchitis:** Secondary consideration for acute inflammatory lower airway disease.
4. **Pleurisy / Pleural Inflammation:** Supported by respiratory discomfort and dyspnea on inspiration.

---

### **Final Diagnosis**
🎯 **Acute Lower Respiratory Infection / Pneumonia**

---

### **Diagnostic Rationale & Explanation**
The diagnostic rationale for **Acute Lower Respiratory Infection / Pneumonia** is derived by synthesizing the patient's presentation ("### Imaging Findings: Atelectasis, Lung Lesion, Pneumonia, Edema, Consolidation") with radiological findings from the uploaded chest radiograph and evidence-based consensus guidelines.

### 📊 **ResNet-50 Multi-Label Pathology Scores**

| Pathology | Probability | Assessment |
| :--- | :--- | :--- |
| **Atelectasis** | **61%** | 🟡 Moderate Risk |
| **Lung Lesion** | **60%** | 🟡 Moderate Risk |
| **Pneumonia** | **59%** | 🟡 Moderate Risk |
| **Edema** | **47%** | 🟡 Moderate Risk |
| **Consolidation** | **43%** | 🟡 Moderate Risk |

### 🫁 **Generated XAI Diagnostic Artifacts & Image Paths**

- 📷 **Original Chest X-ray:** `C:\Users\wwefi\OneDrive\Υπολογιστής\diploma-thesis\ZenithDx_Final\backend\outputs\default\pneumonia_original.png`
- 🔥 **Grad-CAM Overlay:** `C:\Users\wwefi\OneDrive\Υπολογιστής\diploma-thesis\ZenithDx_Final\backend\outputs\default\pneumonia_gradcam_overlay.png`
- 🫁 **Segmented Grad-CAM (S²A-UNet ROI):** `C:\Users\wwefi\OneDrive\Υπολογιστής\diploma-thesis\ZenithDx_Final\backend\outputs\default\pneumonia_gradcam_segmented.png`
- 📊 **Captum Text Attribution Plot:** `C:\Users\wwefi\OneDrive\Υπολογιστής\diploma-thesis\ZenithDx_Final\backend\outputs\default\20260730_120810\captum_query_20260730_120811_seq.png`
- 🧠 **captum_history_seq:** `None`
- 🧠 **captum_history_tok:** `None`
- 🧠 **captum_image_seq:** `C:\Users\wwefi\OneDrive\Υπολογιστής\diploma-thesis\ZenithDx_Final\backend\outputs\default\20260730_120810\captum_image_20260730_120811_seq.png`
- 🧠 **captum_image_tok:** `C:\Users\wwefi\OneDrive\Υπολογιστής\diploma-thesis\ZenithDx_Final\backend\outputs\default\20260730_120810\captum_image_20260730_120811_tok.png`
- 🧠 **captum_query_tok:** `C:\Users\wwefi\OneDrive\Υπολογιστής\diploma-thesis\ZenithDx_Final\backend\outputs\default\20260730_120810\captum_query_20260730_120811_tok.png`

#### **Top Attribution Tokens (PyTorch Captum)**

**Section: Query**
- `chest` (0.818), `pain` (0.805), `93%` (0.650), `inspiration` (0.649), `hypoxia` (0.633)

**Section: Image Findings**
- `lesion` (0.838), `pneumonia` (0.704), `atelectasis` (0.612), `edema` (0.529), `lung` (0.406)

--- 

