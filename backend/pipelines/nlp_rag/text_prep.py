# backend/pipelines/nlp_rag/text_prep.py
from __future__ import annotations

import re
import warnings
from typing import List, Optional

MAX_TOKENS = 2048

# Lazy SciBERT SpaCy loader
_NLP = None

def get_scibert_nlp():
    global _NLP
    if _NLP is None:
        try:
            import en_core_sci_scibert
            _NLP = en_core_sci_scibert.load()
        except Exception:
            try:
                import spacy
                _NLP = spacy.blank("en")
            except Exception:
                _NLP = "fallback"
    return _NLP

def clean_query(text: str) -> str:
    """Preprocesses and normalizes clinical text queries using SciBERT NLP or regex fallback."""
    nlp = get_scibert_nlp()
    if nlp != "fallback":
        try:
            doc = nlp(text.lower())
            tokens = [token.lemma_ for token in doc if getattr(token, "is_alpha", True) and not getattr(token, "is_stop", False)]
            if tokens:
                return " ".join(tokens)
        except Exception as e:
            print(f"[text_prep] SpaCy warning: {e}", file=sys.stderr)
    
    # Regex fallback cleaner if SpaCy is unavailable
    stopwords = {"a", "an", "the", "in", "of", "and", "or", "to", "for", "with", "on", "at", "by", "from", "is", "it", "this", "that"}
    words = re.findall(r'[a-zA-Z]+', text.lower())
    cleaned = [w for w in words if w not in stopwords]
    return " ".join(cleaned) if cleaned else text.lower()

def truncate_prompt_tokens(prompt: str, tokenizer=None, max_tokens: int = MAX_TOKENS) -> str:
    """Truncates prompt to max_tokens using given tokenizer or character estimation fallback."""
    if tokenizer is not None:
        tokens = tokenizer.encode(prompt)
        if len(tokens) > max_tokens:
            truncated = tokenizer.decode(tokens[:max_tokens], skip_special_tokens=True)
            print(f"⚠️ Prompt truncated from {len(tokens)} to {max_tokens} tokens.")
            return truncated
        return prompt
    else:
        # Fallback character estimation if tokenizer is not passed (~4 chars/token)
        max_chars = max_tokens * 4
        if len(prompt) > max_chars:
            return prompt[:max_chars]
        return prompt

def truncate_prompt(prompt: str, limit: int = MAX_TOKENS, tokenizer=None) -> str:
    return truncate_prompt_tokens(prompt, tokenizer=tokenizer, max_tokens=limit)

def chunk_text_by_tokens(text: str, tokenizer, max_tokens: int = MAX_TOKENS) -> List[str]:
    """Splits text into chunks bounded by max_tokens."""
    tokens = tokenizer.encode(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i + max_tokens]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(chunk_text)
    return chunks

def symptom_in_first_sentence(symptom: str, text: str, max_tokens: int = 15) -> bool:
    """Checks if 'symptom' appears in the first max_tokens tokens of the first sentence."""
    symptom = symptom.lower().strip()
    first_sentence = re.split(r'[.!?]\s', text.strip(), maxsplit=1)[0]
    first_tokens = first_sentence.lower().split()[:max_tokens]
    return any(symptom == tok.strip(" ,;:") for tok in first_tokens)
