from pathlib import Path

import pandas as pd

from detector.ingest.join_ell import ELL_MIN_USABLE, build_joined_manifest


def _write_raw(raw_dir: Path, daigt_rows: list[dict], persuade_rows: list[dict]) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(daigt_rows).to_csv(raw_dir / "train_v2_drcat_02.csv", index=False)
    pd.DataFrame(persuade_rows).to_csv(
        raw_dir / "persuade_2.0_human_scores_demo_id_github.csv", index=False
    )


def _daigt_row(text: str, source: str = "persuade_corpus", label: int = 0) -> dict:
    return {
        "text": text,
        "label": label,
        "prompt_name": "Phones and driving",
        "source": source,
        "RDizzl3_seven": False,
    }


def _persuade_row(essay_id: str, text: str, ell_status: str | float) -> dict:
    return {
        "essay_id_comp": essay_id,
        "full_text": text,
        "holistic_essay_score": 3,
        "word_count": len(text.split()),
        "prompt_name": "Phones and driving",
        "task": "Independent",
        "assignment": "Write an essay.",
        "source_text": "",
        "gender": "M",
        "grade_level": 10,
        "ell_status": ell_status,
        "race_ethnicity": "Black/African American",
        "economically_disadvantaged": "",
        "student_disability_status": "",
    }


def test_join_matches_on_normalised_text(tmp_path: Path) -> None:
    daigt = [
        _daigt_row("Hello   world.\nSecond sentence."),
        _daigt_row("This is machine text.", source="chat_gpt_moth", label=1),
    ]
    persuade = [
        _persuade_row("A1", "hello world. second sentence.", "Yes"),
    ]
    _write_raw(tmp_path, daigt, persuade)

    matched, report = build_joined_manifest(tmp_path)

    assert report.daigt_persuade_rows == 1  # only the persuade_corpus row is eligible
    assert report.matched_rows == 1
    assert report.match_rate == 1.0
    assert len(matched) == 1
    assert matched.iloc[0]["ell_status"] == "Yes"


def test_join_counts_ell_status_buckets(tmp_path: Path) -> None:
    daigt = [
        _daigt_row("Essay one text here."),
        _daigt_row("Essay two text here."),
        _daigt_row("Essay three text here."),
    ]
    persuade = [
        _persuade_row("A1", "Essay one text here.", "Yes"),
        _persuade_row("A2", "Essay two text here.", "No"),
        _persuade_row("A3", "Essay three text here.", float("nan")),
    ]
    _write_raw(tmp_path, daigt, persuade)

    _, report = build_joined_manifest(tmp_path)

    assert report.ell_yes_matched == 1
    assert report.ell_no_matched == 1
    assert report.ell_unknown_matched == 1


def test_recommend_ellipse_below_threshold(tmp_path: Path) -> None:
    daigt = [_daigt_row("Only one essay here.")]
    persuade = [_persuade_row("A1", "Only one essay here.", "Yes")]
    _write_raw(tmp_path, daigt, persuade)

    _, report = build_joined_manifest(tmp_path)

    assert report.ell_yes_matched < ELL_MIN_USABLE
    assert report.recommend_ellipse is True


def test_unmatched_daigt_row_is_excluded(tmp_path: Path) -> None:
    daigt = [_daigt_row("This text has no persuade match.")]
    persuade = [_persuade_row("A1", "Completely different text.", "No")]
    _write_raw(tmp_path, daigt, persuade)

    matched, report = build_joined_manifest(tmp_path)

    assert report.matched_rows == 0
    assert report.match_rate == 0.0
    assert len(matched) == 0
