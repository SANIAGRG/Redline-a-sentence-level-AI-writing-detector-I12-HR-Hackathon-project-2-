import pandas as pd

from detector.ingest.manifest import build_manifest
from detector.ingest.sampling import SampledPools


def _pools() -> SampledPools:
    human_baseline = pd.DataFrame(
        {
            "text_hash": ["h1", "h2"],
            "ell_status": ["Yes", None],
            "prompt_name": ["Topic A", "Topic B"],
            "word_count": [300, 400],
        }
    )
    human_training = pd.DataFrame(
        {
            "text_hash": ["h3"],
            "ell_status": ["No"],
            "prompt_name": ["Topic A"],
            "word_count": [350],
        }
    )
    machine_training = pd.DataFrame(
        {"source": ["family_a"], "prompt_name": ["Topic A"], "text": ["some machine text here"]}
    )
    machine_heldout_family = pd.DataFrame(
        {"source": ["falcon_180b_v1"], "prompt_name": ["Topic B"], "text": ["held out text"]}
    )
    return SampledPools(human_baseline, human_training, machine_training, machine_heldout_family)


def test_manifest_row_count_matches_input_pools() -> None:
    manifest = build_manifest(_pools())
    assert len(manifest) == 5  # 2 + 1 + 1 + 1


def test_manifest_doc_ids_unique() -> None:
    manifest = build_manifest(_pools())
    assert manifest["doc_id"].is_unique


def test_manifest_ell_flag_fills_missing_as_unknown() -> None:
    manifest = build_manifest(_pools())
    baseline_rows = manifest[manifest["pool"] == "human_baseline"]
    assert set(baseline_rows["ell_flag"]) == {"Yes", "unknown"}


def test_manifest_author_type_split() -> None:
    manifest = build_manifest(_pools())
    assert (manifest["author_type"] == "human").sum() == 3
    assert (manifest["author_type"] == "machine").sum() == 2


def test_manifest_machine_word_count_derived_from_text() -> None:
    manifest = build_manifest(_pools())
    row = manifest[manifest["source"] == "family_a"].iloc[0]
    assert row["word_count"] == 4  # "some machine text here"


def test_manifest_required_columns_present() -> None:
    manifest = build_manifest(_pools())
    required = {
        "source",
        "author_type",
        "generator",
        "prompt_condition",
        "ell_flag",
        "topic",
        "word_count",
        "license",
    }
    assert required.issubset(manifest.columns)
