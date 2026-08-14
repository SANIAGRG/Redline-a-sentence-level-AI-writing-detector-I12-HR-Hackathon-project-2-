from detector.generate.adversarial import inject_typos, paraphrase_synonyms


def test_inject_typos_zero_rate_leaves_text_unchanged() -> None:
    text = "The government should not allow phones in class."
    assert inject_typos(text, rate=0.0, seed=1) == text


def test_inject_typos_changes_text_at_high_rate() -> None:
    text = "The government should not allow phones during class today."
    result = inject_typos(text, rate=1.0, seed=1)
    assert result != text


def test_inject_typos_preserves_word_count() -> None:
    text = "The government should not allow phones during class today."
    result = inject_typos(text, rate=0.5, seed=1)
    assert len(result.split(" ")) == len(text.split(" "))


def test_inject_typos_deterministic_for_fixed_seed() -> None:
    text = "The government should not allow phones during class today."
    a = inject_typos(text, rate=0.5, seed=7)
    b = inject_typos(text, rate=0.5, seed=7)
    assert a == b


def test_inject_typos_leaves_short_words_untouched() -> None:
    # words under 3 chars are never perturbed
    text = "a it is ok"
    result = inject_typos(text, rate=1.0, seed=1)
    assert result == text


def test_paraphrase_synonyms_zero_rate_leaves_text_unchanged() -> None:
    text = "The government should not allow students to use phones."
    assert paraphrase_synonyms(text, rate=0.0, seed=1) == text


def test_paraphrase_synonyms_changes_some_content_words_at_high_rate() -> None:
    text = "The government should not allow students to use phones during class."
    result = paraphrase_synonyms(text, rate=1.0, seed=1)
    assert result != text


def test_paraphrase_synonyms_deterministic_for_fixed_seed() -> None:
    text = "The government should not allow students to use phones during class."
    a = paraphrase_synonyms(text, rate=0.5, seed=3)
    b = paraphrase_synonyms(text, rate=0.5, seed=3)
    assert a == b


def test_paraphrase_synonyms_preserves_approximate_length() -> None:
    text = "The government should not allow students to use phones during class."
    result = paraphrase_synonyms(text, rate=0.5, seed=3)
    # word-for-word substitution, so word count should match closely
    assert abs(len(result.split()) - len(text.split())) <= 2
