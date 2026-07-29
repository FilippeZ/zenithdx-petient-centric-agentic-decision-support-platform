import json
from nltk.tokenize import word_tokenize
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import nltk
import evaluate
import spacy

nltk.download('punkt')

# ----- Medical NER model (scispaCy) -----
nlp = spacy.load("en_ner_bc5cdr_md")  # ή "en_core_sci_sm" για πιο γενικό NER

def extract_medical_concepts(text):
    doc = nlp(text)
    return set(ent.text.lower() for ent in doc.ents if ent.label_ in {"DISEASE", "CHEMICAL"})

def medical_concept_recall(pred, gold):
    gold_concepts = extract_medical_concepts(gold)
    pred_concepts = extract_medical_concepts(pred)
    if not gold_concepts:
        return 1.0
    recall = len(gold_concepts & pred_concepts) / len(gold_concepts)
    return recall

# ----- FILE PATHS -----
GROUND_TRUTH_FILE = "alpaca_style_output_pharm.jsonl"   # JSONL with 'output'
PREDICTIONS_FILE = "pharm_predictions.json"             # JSON array

# Load ground truth JSONL (list of dicts)
ground_truths_json = []
with open(GROUND_TRUTH_FILE, "r", encoding="utf-8") as fin:
    for line in fin:
        if line.strip():
            ground_truths_json.append(json.loads(line))

# Load predictions JSON (list of dicts)
with open(PREDICTIONS_FILE, "r", encoding="utf-8") as fin:
    predictions_json = json.load(fin)

# Match last 30 samples (by order)
ground_truths_json = ground_truths_json[-30:]
predictions_json = predictions_json[-30:]

# Extract the correct outputs
ground_truths = [sample["output"].strip() for sample in ground_truths_json]
predictions   = [sample["therapeutic_report_prediction"].strip() for sample in predictions_json]

assert len(ground_truths) == len(predictions), f"Mismatch: {len(ground_truths)} GT, {len(predictions)} predictions!"

# ----- TOKEN-LEVEL F1 -----
def token_f1(pred, gold):
    try:
        pred_tokens = set(word_tokenize(pred.lower()))
        gold_tokens = set(word_tokenize(gold.lower()))
    except LookupError:
        pred_tokens = set(pred.lower().split())
        gold_tokens = set(gold.lower().split())
    tp = len(pred_tokens & gold_tokens)
    if tp == 0:
        return 0.0
    precision = tp / len(pred_tokens) if pred_tokens else 0
    recall = tp / len(gold_tokens) if gold_tokens else 0
    if precision + recall == 0:
        return 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return f1

f1s = [token_f1(p, g) for p, g in zip(predictions, ground_truths)]
print("Mean Token-level F1: {:.4f}".format(sum(f1s) / len(f1s)))

# ----- ROUGE-L -----
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
rouge_ls = [scorer.score(g, p)['rougeL'].fmeasure for p, g in zip(predictions, ground_truths)]
print("Mean ROUGE-L F1: {:.4f}".format(sum(rouge_ls) / len(rouge_ls)))

# ----- BERTScore -----
P, R, F1 = bert_score(predictions, ground_truths, lang='en', rescale_with_baseline=True)
print("Mean BERTScore F1: {:.4f}".format(float(F1.mean())))

# ----- BLEU -----
bleu = evaluate.load("bleu")
bleu_score = bleu.compute(predictions=predictions, references=[[gt] for gt in ground_truths])
print("BLEU Score: {:.4f}".format(bleu_score["bleu"]))

# ----- Medical Concept Recall -----
concept_recalls = [medical_concept_recall(p, g) for p, g in zip(predictions, ground_truths)]
print("Mean Medical Concept Recall: {:.4f}".format(sum(concept_recalls) / len(concept_recalls)))

# ----- Per-sample (optional) -----
show_examples = False
if show_examples:
    for i, (gt, pred) in enumerate(zip(ground_truths, predictions)):
        print(f"\nSample {i+1}")
        print("-" * 60)
        print("GROUND TRUTH:\n", gt)
        print("PREDICTION:\n", pred)
        print("Token F1: {:.4f}".format(token_f1(pred, gt)))
        print("ROUGE-L: {:.4f}".format(scorer.score(gt, pred)['rougeL'].fmeasure))
        print("BERTScore: {:.4f}".format(F1[i]))
        print("Medical Concept Recall: {:.4f}".format(medical_concept_recall(pred, gt)))
        print("-" * 60)
