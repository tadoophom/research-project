"""Clinical sentiment analysis utilities.

This module extends the existing :class:`KeywordContextAnalyzer` with
medical specific functionality such as simple de-identification,
entity extraction and negation detection.  The implementation is a
light‑weight prototype aimed at demonstrating how the project could be
extended for clinical text.  It attempts to use optional packages
``scispacy`` and ``negspacy`` when available.  When these dependencies
cannot be installed (for example in restricted environments) the code
falls back to the standard spaCy model and disables the optional
features so that the rest of the pipeline can still execute.

The design follows the high level specification given in the project
brief and showcases how UV managed dependencies can be leveraged in
code.  Advanced features such as ontology linking or temporal reasoning
are intentionally left as future work.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Dict, Any

import pandas as pd
import spacy

from keyword_context_analyzer import KeywordContextAnalyzer

try:  # pragma: no cover - optional dependency
    from negspacy.negation import Negex
    _HAS_NEGSPACY = True
except Exception:  # noqa: BLE001 - broad to keep optional
    _HAS_NEGSPACY = False


@dataclass
class MedicalTextPreprocessor:
    """Basic preprocessor performing very small scale de‑identification.

    The routine replaces names (capitalised words) and long numbers with
    generic placeholders.  This is **not** a production ready
    de‑identification algorithm but serves as a placeholder to
    demonstrate where such logic would live in the system.
    """

    name_placeholder: str = "PATIENT"

    def deidentify(self, text: str) -> str:
        text = re.sub(r"\b[A-Z][a-z]+\b", self.name_placeholder, text)
        text = re.sub(r"\d{2,}", "0", text)
        return text


class ClinicalSentimentAnalyzer(KeywordContextAnalyzer):
    """Analyzer with clinical specific extensions.

    Parameters
    ----------
    model_name:
        spaCy model to load.  ``"en_core_sci_sm"`` is attempted first and
        the method falls back to ``"en_core_web_sm"`` when the scientific
        model is unavailable.  This keeps the example runnable even when
        the larger biomedical model cannot be downloaded.
    """

    def __init__(self, model_name: str = "en_core_sci_sm") -> None:
        try:
            super().__init__(model_name)
        except Exception:  # noqa: BLE001 - model might be missing
            super().__init__("en_core_web_sm")

        self.preprocessor = MedicalTextPreprocessor()

        if _HAS_NEGSPACY and "negex" not in self.nlp.pipe_names:
            negex = Negex(self.nlp, language="en")
            self.nlp.add_pipe(negex, last=True)

    # -----------------------------------------------------------------
    # Core analysis API
    # -----------------------------------------------------------------
    def analyze_documents(self, texts: Iterable[str], keywords: Iterable[str]) -> pd.DataFrame:
        """Analyze a collection of documents for keyword sentiment.

        Each document is de‑identified, processed with spaCy and then the
        inherited keyword context logic is used to obtain sentiment
        scores.  Additional columns provide information about recognised
        entities and whether the keyword occurrence was negated.
        """

        results: List[Dict[str, Any]] = []
        keywords_lower = [k.lower() for k in keywords]

        for text_id, raw_text in enumerate(texts):
            clean_text = self.preprocessor.deidentify(raw_text)
            doc = self.nlp(clean_text)

            for token in doc:
                if token.text.lower() in keywords_lower:
                    sentiment = self.analyze_context_sentiment(
                        self.get_dependency_context(doc, token)
                    )

                    entities = [
                        f"{ent.text}:{ent.label_}"
                        for ent in doc.ents
                        if ent.start <= token.i < ent.end
                    ]

                    results.append(
                        {
                            "text_id": text_id,
                            "sentence": token.sent.text,
                            "keyword": token.text,
                            "negated": bool(getattr(token._, "negex", False)),
                            "entities": ", ".join(entities),
                            "sentiment": sentiment["combined"],
                        }
                    )

        return pd.DataFrame(results)


def demo() -> None:
    """Small demonstration executed when running the module directly."""

    texts = [
        "Patient responded excellently to chemotherapy protocol.",
        "No evidence of infection was detected in the latest tests.",
    ]

    analyzer = ClinicalSentimentAnalyzer()
    df = analyzer.analyze_documents(texts, ["chemotherapy", "infection"])
    print(df)


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    demo()

