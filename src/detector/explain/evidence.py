"""Evidence assembly (Module 6A): given a pasted essay, returns the
document-level calibrated probability + uncertainty band (or abstain),
a per-sentence heat-map signal, and a plain-language evidence panel.
Tested independently of the UI -- the app is a thin rendering layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from detector.config import RedlineConfig, load_config
from detector.features.likelihood import _all_sentences, compute_document_likelihood
from detector.features.stylometric import compute_document_features
from detector.model.build_training_frame import (
    ALL_FEATURE_COLS,
    LIKELIHOOD_COLS,
    STYLOMETRIC_COLS,
    Z_COLS,
    build_zscored_frame,
)

MIN_WORDS_FOR_VERDICT = 150

FEATURE_DESCRIPTIONS: dict[str, str] = {
    "sent_len_mean": "Average sentence length (words)",
    "sent_len_std": "Sentence-length variation",
    "sent_len_burstiness": "Rhythmic evenness of sentence length (machine prose is flatter)",
    "mattr": "Vocabulary diversity (moving-average type-token ratio)",
    "hapax_rate": "Rate of words used only once",
    "function_word_rate": "Rate of function words (the, of, and, ...)",
    "pos_trigram_entropy": "Predictability of grammatical structure",
    "clause_count_mean": "Average clauses per sentence",
    "subordination_ratio_mean": "Rate of subordinate clauses",
    "em_dash_rate": "Em-dash usage rate",
    "semicolon_rate": "Semicolon usage rate",
    "oxford_comma_rate": "Oxford-comma consistency",
    "construction_template_rate": "Rate of stock constructions (\"not just X, but Y\", rule-of-three, etc.)",
    "discourse_marker_rate": "Rate of discourse markers (moreover, furthermore, ...)",
    "spelling_error_rate": "Spelling-error rate",
    "homophone_error_rate": "Rate of common native-English error patterns (could of, etc.)",
    "proper_noun_rate": "Proper-noun density",
    "numeral_rate": "Numeral density",
    "named_entity_rate": "Named-entity density",
    "mean_logprob": "Average model log-probability of the observed words",
    "mean_lograsnk": "Average rank of observed words in the model's predicted distribution",
    "lrr": "Log-likelihood / log-rank ratio",
    "mean_entropy": "Model's uncertainty at each word position",
    "mean_curvature": "Conditional probability curvature (Fast-DetectGPT signal)",
    "logprob_variance": "Variability of word-level surprise within the sentence",
    "cross_ppl_ratio": "Cross-model perplexity ratio (Binoculars signal; reduces ESL false positives)",
}


@dataclass
class SentenceEvidence:
    text: str
    paragraph_index: int
    sentence_index: int
    signal: float  # higher = more machine-like, relative z-scored curvature+logprob


@dataclass
class FeatureEvidenceRow:
    name: str
    description: str
    value: float
    baseline_mean: float
    z_score: float
    contribution: float


@dataclass
class EssayAnalysis:
    word_count: int
    abstained: bool
    probability: float | None = None
    uncertainty_band: tuple[float, float] | None = None
    sentence_evidence: list[SentenceEvidence] = field(default_factory=list)
    feature_evidence: list[FeatureEvidenceRow] = field(default_factory=list)
    abstain_reason: str | None = None


_CACHE: dict = {}


def _load_bundle(config: RedlineConfig) -> dict:
    if "bundle" not in _CACHE:
        model_path = config.paths.features.parent / "model" / "logistic_regression.joblib"
        _CACHE["bundle"] = joblib.load(model_path)
    return _CACHE["bundle"]


def _load_baseline_stats(config: RedlineConfig) -> pd.DataFrame:
    if "baseline_stats" not in _CACHE:
        from detector.features.corpus_relative import build_baseline_stats

        frame = build_zscored_frame(config)  # also warms full frame cache below
        _CACHE["full_frame"] = frame
        baseline = frame[frame["pool"] == "baseline"]
        _CACHE["baseline_stats"] = build_baseline_stats(baseline, ALL_FEATURE_COLS)
        _CACHE["baseline_means"] = baseline[ALL_FEATURE_COLS].mean()
    return _CACHE["baseline_stats"]


def _plain_coefficients(config: RedlineConfig) -> np.ndarray:
    """A plain (uncalibrated) LogisticRegression fit on the same data,
    purely so we have interpretable coefficients for the evidence panel
    -- CalibratedClassifierCV wraps the estimator, making coef_ access
    awkward under time pressure. The calibrated model still produces the
    actual probability shown to the user.
    """
    if "coefs" not in _CACHE:
        frame = _CACHE.get("full_frame")
        if frame is None:
            frame = build_zscored_frame(config)
        trainable = frame[frame["label"].isin([0, 1])]
        X = trainable[Z_COLS].fillna(0.0).values
        y = trainable["label"].values
        lr = LogisticRegression(penalty="l2", C=1.0, max_iter=1000)
        lr.fit(X, y)
        _CACHE["coefs"] = lr.coef_[0]
    return _CACHE["coefs"]


def analyze_essay(text: str, config: RedlineConfig | None = None) -> EssayAnalysis:
    config = config or load_config()
    word_count = len(text.split())

    if word_count < MIN_WORDS_FOR_VERDICT:
        return EssayAnalysis(
            word_count=word_count,
            abstained=True,
            abstain_reason=f"Essay is under {MIN_WORDS_FOR_VERDICT} words -- too short for a reliable verdict.",
        )

    bundle = _load_bundle(config)
    baseline_stats = _load_baseline_stats(config)
    baseline_means = _CACHE["baseline_means"]
    coefs = _plain_coefficients(config)

    stylo = compute_document_features("live", text)
    stylo_dict = stylo.__dict__

    all_sents = _all_sentences(text)
    likelihood_results = compute_document_likelihood("live", text, candidates=all_sents)
    if not likelihood_results:
        return EssayAnalysis(word_count=word_count, abstained=True, abstain_reason="Could not segment essay.")

    likelihood_means = {
        col: float(np.mean([getattr(r, col) for r in likelihood_results])) for col in LIKELIHOOD_COLS
    }

    row = {**{c: stylo_dict[c] for c in STYLOMETRIC_COLS}, **likelihood_means}

    from detector.features.corpus_relative import length_band

    band = length_band(word_count)
    z_values = {}
    for col in ALL_FEATURE_COLS:
        mean_key, std_key = f"{col}__mean", f"{col}__std"
        if band in baseline_stats.index and std_key in baseline_stats.columns:
            mean = baseline_stats.loc[band, mean_key]
            std = baseline_stats.loc[band, std_key]
            if not std or np.isnan(std):
                mean, std = baseline_means[col], 1.0
        else:
            mean, std = baseline_means[col], 1.0
        z_values[col] = (row[col] - mean) / std if std else 0.0

    z_vec = np.array([z_values[c] for c in ALL_FEATURE_COLS]).reshape(1, -1)
    probability = float(bundle["model"].predict_proba(z_vec)[0, 1])
    band_width = 0.15 * (1 - abs(probability - 0.5) * 2) + 0.05
    uncertainty = (max(0.0, probability - band_width), min(1.0, probability + band_width))

    sentence_evidence = [
        SentenceEvidence(
            text=r.text,
            paragraph_index=r.paragraph_index,
            sentence_index=r.sentence_index,
            signal=r.mean_curvature - r.mean_logprob,
        )
        for r in likelihood_results
    ]

    contributions = z_vec[0] * coefs
    feature_evidence = [
        FeatureEvidenceRow(
            name=col,
            description=FEATURE_DESCRIPTIONS.get(col, col),
            value=float(row[col]),
            baseline_mean=float(baseline_means[col]),
            z_score=float(z_values[col]),
            contribution=float(contributions[i]),
        )
        for i, col in enumerate(ALL_FEATURE_COLS)
    ]
    feature_evidence.sort(key=lambda r: abs(r.contribution), reverse=True)

    return EssayAnalysis(
        word_count=word_count,
        abstained=False,
        probability=probability,
        uncertainty_band=uncertainty,
        sentence_evidence=sentence_evidence,
        feature_evidence=feature_evidence,
    )
