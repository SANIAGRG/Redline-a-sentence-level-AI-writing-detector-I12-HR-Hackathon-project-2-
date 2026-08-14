import pandas as pd

from detector.generate.run_modern import (
    CONDITIONS,
    INDEPENDENT_PROMPTS,
    MODELS,
    N_PER_MODEL,
    build_prompt,
    build_targets,
    pick_exemplar,
)


def test_build_targets_covers_every_model_n_times() -> None:
    targets = build_targets(seed=1)
    assert len(targets) == len(MODELS) * N_PER_MODEL
    for model in MODELS:
        assert sum(1 for t in targets if t["model"] == model) == N_PER_MODEL


def test_build_targets_essay_ids_unique() -> None:
    targets = build_targets(seed=1)
    ids = [t["essay_id"] for t in targets]
    assert len(ids) == len(set(ids))


def test_build_targets_covers_all_conditions_per_model() -> None:
    targets = build_targets(seed=1)
    for model in MODELS:
        model_conditions = {t["condition"] for t in targets if t["model"] == model}
        assert model_conditions == set(CONDITIONS)


def test_build_targets_covers_all_topics_per_model() -> None:
    targets = build_targets(seed=1)
    for model in MODELS:
        model_topics = {t["topic"] for t in targets if t["model"] == model}
        assert model_topics == set(INDEPENDENT_PROMPTS)


def test_build_targets_deterministic_for_fixed_seed() -> None:
    a = build_targets(seed=42)
    b = build_targets(seed=42)
    assert a == b


def test_build_prompt_bare_has_no_persona_or_exemplar_text() -> None:
    prompt = build_prompt("Write about phones.", "bare", None)
    assert "17-year-old" not in prompt
    assert "example of a well-written" not in prompt
    assert "Write about phones." in prompt


def test_build_prompt_persona_includes_persona_framing() -> None:
    prompt = build_prompt("Write about phones.", "persona", None)
    assert "17-year-old" in prompt
    assert "Write about phones." in prompt


def test_build_prompt_style_exemplar_includes_exemplar_text() -> None:
    prompt = build_prompt("Write about phones.", "style_exemplar", "Some human essay text.")
    assert "Some human essay text." in prompt
    assert "Write about phones." in prompt


def test_build_prompt_evasion_includes_evasion_instruction() -> None:
    prompt = build_prompt("Write about phones.", "evasion", None)
    assert "avoid" in prompt.lower()
    assert "Write about phones." in prompt


def test_pick_exemplar_prefers_same_topic() -> None:
    joined = pd.DataFrame(
        {
            "prompt_name": ["Topic A", "Topic A", "Topic B"],
            "text": ["word " * 200, "different " * 200, "other " * 200],
        }
    )
    excerpt = pick_exemplar(joined, "Topic B", seed=1)
    assert excerpt.startswith("other")


def test_pick_exemplar_falls_back_when_topic_missing() -> None:
    joined = pd.DataFrame({"prompt_name": ["Topic A"], "text": ["word " * 200]})
    excerpt = pick_exemplar(joined, "Nonexistent Topic", seed=1)
    assert excerpt.startswith("word")


def test_pick_exemplar_truncates_to_150_words() -> None:
    joined = pd.DataFrame({"prompt_name": ["Topic A"], "text": ["word " * 300]})
    excerpt = pick_exemplar(joined, "Topic A", seed=1)
    assert len(excerpt.split()) == 150
