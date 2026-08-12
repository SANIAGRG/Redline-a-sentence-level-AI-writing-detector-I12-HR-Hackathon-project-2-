from detector.generate.run_polish import (
    INTENSITIES,
    assign_intensities,
    polish_essay,
    split_paragraphs,
)


class FakeClient:
    """Stands in for OllamaClient so tests never touch a real server."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def generate(self, model: str, prompt: str) -> str:
        self.calls.append((model, prompt))
        return f"REVISED::{prompt[-20:]}"


def test_split_paragraphs_on_blank_lines() -> None:
    text = "Para one.\nStill para one.\n\nPara two.\n\n\nPara three."
    assert split_paragraphs(text) == ["Para one.\nStill para one.", "Para two.", "Para three."]


def test_split_paragraphs_drops_empty_chunks() -> None:
    assert split_paragraphs("Only one paragraph, no blank lines.") == [
        "Only one paragraph, no blank lines."
    ]


def test_assign_intensities_covers_all_three_for_large_n() -> None:
    assigned = assign_intensities(30, seed=1)
    assert len(assigned) == 30
    assert set(assigned) == set(INTENSITIES)


def test_assign_intensities_deterministic_for_fixed_seed() -> None:
    a = assign_intensities(20, seed=7)
    b = assign_intensities(20, seed=7)
    assert a == b


def test_polish_essay_leaves_some_paragraphs_untouched() -> None:
    text = "P1 sentence.\n\nP2 sentence.\n\nP3 sentence.\n\nP4 sentence.\n\nP5 sentence."
    client = FakeClient()

    results = polish_essay(client, text, intensity="flow", seed=42)

    assert len(results) == 5
    revised_flags = [r.was_revised for r in results]
    # revision probability is 40-100%, so at least one but not necessarily all
    assert any(revised_flags)
    for r in results:
        if r.was_revised:
            assert r.revised_text.startswith("REVISED::")
        else:
            assert r.revised_text == r.original_text


def test_polish_essay_only_calls_model_for_revised_paragraphs() -> None:
    text = "P1.\n\nP2.\n\nP3."
    client = FakeClient()

    results = polish_essay(client, text, intensity="grammar", seed=1)

    n_revised = sum(r.was_revised for r in results)
    assert len(client.calls) == n_revised
