from detector.generate.text_cleanup import strip_llm_preamble


def test_strips_heres_a_revised_version_preamble() -> None:
    text = "Here's a revised version of the paragraph:\n\nThe actual content starts here."
    assert strip_llm_preamble(text) == "The actual content starts here."


def test_strips_here_is_the_revised_paragraph_preamble() -> None:
    text = "Here is the revised paragraph:\n\nContent follows."
    assert strip_llm_preamble(text) == "Content follows."


def test_strips_sure_preamble() -> None:
    text = "Sure! Here is the paragraph:\n\nContent follows."
    assert strip_llm_preamble(text) == "Content follows."


def test_leaves_clean_text_unchanged() -> None:
    text = "This paragraph has no preamble at all and should pass through untouched."
    assert strip_llm_preamble(text) == text


def test_does_not_strip_when_no_blank_line_separator() -> None:
    # "Here" appearing naturally in real content, with no preamble structure
    text = "Here in this town, life moves slowly."
    assert strip_llm_preamble(text) == text


def test_preserves_internal_blank_lines_in_clean_content() -> None:
    text = "First sentence.\n\nSecond sentence, still real content."
    assert strip_llm_preamble(text) == text
