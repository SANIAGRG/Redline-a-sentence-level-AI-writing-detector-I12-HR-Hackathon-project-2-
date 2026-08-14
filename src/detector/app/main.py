"""Redline -- a sentence-level AI-writing detector that shows its
evidence. Module 6B. Thin rendering layer over
src/detector/explain/evidence.py -- no scoring logic here.

Sample-essay picker skipped under the deadline (ADR 0008, spec's own
cut list).
"""

from __future__ import annotations

from typing import Any, cast

import gradio as gr

from detector.explain.evidence import EssayAnalysis, analyze_essay

NOTICE = (
    "**Not suitable for disciplinary or admissions decisions.** Redline shows evidence "
    "for human judgement -- it does not replace it."
)

FOOTER = (
    "Reference corpus: PERSUADE 2.0 (US grades 6-12 persuasive essays) + DAIGT-V2 "
    "2023-era machine essays, small deadline-scoped samples (see docs/LIMITATIONS.md). "
    "Coverage gap: not real college admissions essays; small sample sizes throughout."
)


def _heat_html(analysis: EssayAnalysis) -> str:
    if not analysis.sentence_evidence:
        return "<p><em>No sentences scored.</em></p>"
    signals = [s.signal for s in analysis.sentence_evidence]
    lo, hi = min(signals), max(signals)
    spread = (hi - lo) or 1.0

    parts = []
    for s in analysis.sentence_evidence:
        t = (s.signal - lo) / spread
        r = int(255 * t)
        g = int(255 * (1 - t))
        color = f"rgba({r},{g},80,0.35)"
        parts.append(f'<span style="background-color:{color}; padding:2px;">{s.text}</span>')
    return "<p>" + " ".join(parts) + "</p>"


def _evidence_table(analysis: EssayAnalysis) -> str:
    if analysis.abstained:
        return ""
    rows = analysis.feature_evidence[:10]
    lines = [
        "| Signal | Value | Baseline | Z-score | Contribution |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        direction = "toward machine" if r.contribution > 0 else "toward human"
        lines.append(
            f"| {r.description} | {r.value:.3f} | {r.baseline_mean:.3f} | {r.z_score:+.2f} | "
            f"{direction} ({r.contribution:+.3f}) |"
        )
    return "\n".join(lines)


def run_analysis(text: str) -> tuple[str, str, str]:
    if not text or not text.strip():
        return "Paste an essay first.", "", ""

    analysis = analyze_essay(text)

    if analysis.abstained:
        header = f"**Abstained.** {analysis.abstain_reason} (word count: {analysis.word_count})"
        return header, "", ""

    assert analysis.uncertainty_band is not None
    lo, hi = analysis.uncertainty_band
    header = (
        f"**Document probability (machine-influenced): {analysis.probability:.0%}** "
        f"(uncertainty band: {lo:.0%}-{hi:.0%}) -- word count: {analysis.word_count}"
    )
    return header, _heat_html(analysis), _evidence_table(analysis)


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Redline") as demo:
        gr.Markdown("# Redline")
        gr.Markdown("*a sentence-level AI-writing detector that shows its evidence*")
        gr.Markdown(NOTICE)

        with gr.Row():
            with gr.Column(scale=1):
                text_input = gr.Textbox(label="Paste an essay", lines=20)
                analyze_btn = gr.Button("Analyse", variant="primary")
            with gr.Column(scale=1):
                header_out = gr.Markdown()
                heat_out = gr.HTML()
                evidence_out = gr.Markdown()

        # Gradio's type stubs inconsistently expose Button.click across
        # versions (unpinned dependency) -- cast to Any rather than an
        # ignore-comment, which would flip between needed and unused
        # depending on which gradio version happens to install.
        cast(Any, analyze_btn).click(
            fn=run_analysis, inputs=[text_input], outputs=[header_out, heat_out, evidence_out]
        )

        gr.Markdown("---")
        gr.Markdown(FOOTER)

    return demo


def main() -> None:
    app = build_app()
    app.launch()


if __name__ == "__main__":
    main()
