from __future__ import annotations

import json
from typing import Any, Generator
import requests

import pandas as pd

from trajsem.schemas import SemanticEvent
from trajsem.report import make_markdown_report, summarize_metrics, _label

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

SYSTEM_RULES = """\
You are TrajSem, an evidence-grounded MD (molecular dynamics) trajectory \
analysis assistant.  You are given a structured summary of a protein-ligand \
MD simulation that has already been extracted and analysed.

Rules you MUST follow:
- Only make claims supported by the supplied metrics and semantic events.
- Do NOT infer thermodynamic stability, causality, allostery, or binding \
  affinity unless the evidence explicitly supports it.
- For every mechanistic sentence, cite the relevant event tag, e.g. [E1].
- If the user asks about something not covered by the evidence, say exactly \
  what data is missing.
- State limitations clearly.
- Never output placeholder text or ask the user for column headers — all \
  the data you need is provided below in plain English."""

MANUSCRIPT_SYSTEM_RULES = """\
You are a senior computational biophysicist and scientific writer specialising \
in molecular dynamics simulations and protein-ligand interactions. \
You are given a fully extracted, machine-verified set of trajectory metrics \
and semantic events from a TrajSem analysis pipeline.

Your task is to write a COMPLETE, PUBLICATION-READY scientific manuscript in \
academic third-person prose, formatted in Markdown. Follow the IMRaD structure \
exactly as specified.

Non-negotiable rules:
1. Every mechanistic claim MUST cite the relevant semantic event tag, e.g. [E1].
2. Do NOT fabricate, speculate, or infer data not present in the supplied \
   evidence block.
3. Do NOT make thermodynamic free-energy, entropic, or binding-affinity claims \
   unless the numeric evidence explicitly supports them.
4. Use precise scientific language: Å, ns, occupancy fractions, mean ± SD \
   where available.
5. Write a COMPLETE manuscript — do not truncate or summarise sections. \
   Every section must be fully written out.
6. Do NOT output placeholder text, ellipses, or "see above" references.
7. The Limitations section must be honest and specific to the data provided."""


# ---------------------------------------------------------------------------
# Evidence-to-text renderers (shared between all prompt builders)
# ---------------------------------------------------------------------------

def _metrics_to_text(metrics: dict[str, Any]) -> str:
    """Render the metrics dict as a readable paragraph + bullet list."""
    parts: list[str] = []
    frames = metrics.get("frames", "unknown")
    duration = metrics.get("duration_ns", "unknown")
    parts.append(f"The trajectory contains {frames} frames spanning {duration} ns.")
    parts.append("")
    parts.append("Descriptor statistics (mean / median / min / max):")
    for k, v in metrics.items():
        if isinstance(v, dict) and {"mean", "median", "min", "max"} <= v.keys():
            name = _label(k)
            parts.append(
                f"  - {name}: {v['mean']} / {v['median']} / {v['min']} / {v['max']}"
            )
    return "\n".join(parts)


def _events_to_text(events: list[SemanticEvent]) -> str:
    """Render the event list as numbered human-readable entries."""
    if not events:
        return "No semantic events were detected with the current thresholds."
    parts: list[str] = []
    for i, ev in enumerate(events, 1):
        t0, t1 = ev.time_window_ns
        parts.append(f"[E{i}] {ev.title}")
        parts.append(f"  Type: {ev.event_type}")
        parts.append(f"  Time window: {t0:.3f} – {t1:.3f} ns")
        parts.append(f"  Confidence: {ev.confidence:.2f}  |  Severity: {ev.severity}")
        if ev.entities:
            ent_str = ", ".join(f"{k}={v}" for k, v in ev.entities.items())
            parts.append(f"  Entities: {ent_str}")
        if ev.evidence:
            for ek, ev_val in ev.evidence.items():
                val_str = (
                    ", ".join(str(x) for x in ev_val)
                    if isinstance(ev_val, list)
                    else str(ev_val)
                )
                parts.append(f"  Evidence — {ek}: {val_str}")
        parts.append(f"  Interpretation: {ev.interpretation}")
        parts.append("")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def build_grounded_prompt(
    metrics: dict[str, Any], events: list[SemanticEvent], project_name: str
) -> str:
    context = (
        f"PROJECT: {project_name}\n\n"
        f"=== TRAJECTORY METRICS ===\n{_metrics_to_text(metrics)}\n\n"
        f"=== DETECTED SEMANTIC EVENTS ===\n{_events_to_text(events)}\n\n"
        "=== TASK ===\n"
        "Write a Markdown report with these sections:\n"
        "1. Overview — trajectory summary, stability assessment, key descriptor ranges.\n"
        "2. Features — per-descriptor statistics.\n"
        "3. Semantic Events — each event with evidence, time window, interpretation. "
        "Cite event tags [E1], [E2], etc.\n"
        "4. Limitations.\n\n"
        "Do NOT use placeholder text. Do NOT ask for additional data. "
        "Every claim must be grounded in the evidence above."
    )
    return SYSTEM_RULES + "\n\n" + context


def build_manuscript_prompt(
    metrics: dict[str, Any], events: list[SemanticEvent], project_name: str
) -> str:
    """
    Build a prompt for a Results + Discussion scientific report section.
    Strictly grounded in the structured metrics dict and semantic event list —
    no introduction, abstract, or speculative writing.
    """
    n_events = len(events)
    frames = metrics.get("frames", "N/A")
    duration = metrics.get("duration_ns", "N/A")
    event_tags = (
        ", ".join(f"[E{i}]" for i in range(1, n_events + 1))
        if n_events
        else "none detected"
    )

    # Enumerate only the descriptor keys that actually have data
    available_descriptors = [
        f"  - {_label(k)}: mean={v['mean']}, median={v['median']}, "
        f"min={v['min']}, max={v['max']}"
        for k, v in metrics.items()
        if isinstance(v, dict) and {"mean", "median", "min", "max"} <= v.keys()
    ]
    descriptor_list = "\n".join(available_descriptors) if available_descriptors else "  (none)"

    context = (
        f"PROJECT: {project_name}\n"
        f"Frames: {frames}  |  Duration: {duration} ns  |  Events detected: {n_events}\n\n"
        f"=== AVAILABLE DESCRIPTOR STATISTICS ===\n{descriptor_list}\n\n"
        f"=== SEMANTIC EVENTS ({n_events} total, cite as [E1]…[E{n_events}]) ===\n"
        f"{_events_to_text(events)}\n\n"
        "=== WRITING INSTRUCTIONS ===\n\n"
        "Write ONLY the Results and Discussion sections of a scientific manuscript in Markdown.\n"
        "Do not write Abstract, Introduction, Methods, Conclusions, or any other section.\n\n"
        "Rules:\n"
        "1. Use ONLY the numeric values listed above — do not invent or estimate any data.\n"
        "2. Cite every event by its tag [E1], [E2], etc. in the relevant paragraph.\n"
        "3. Do NOT speculate beyond the evidence (no free-energy, entropic, or causal claims "
        "unless the numbers explicitly support them).\n"
        "4. Write in concise, third-person academic prose.\n"
        "5. Only include sub-sections for descriptors that are actually present in the data above.\n\n"
        "---\n\n"
        f"# Results and Discussion — {project_name}\n\n"
        "## Results\n\n"
        "### Trajectory Stability\n"
        "Report protein RMSD and radius of gyration using the exact mean/median/min/max values "
        "above. One paragraph only.\n\n"
        "### Ligand Binding Pose\n"
        "Report ligand RMSD, pocket volume, and binding-site SASA from the exact values above. "
        "One paragraph only. Omit this section if none of these descriptors are present.\n\n"
        "### Non-covalent Interaction Network\n"
        "Report H-bond count, salt bridges, water bridges, and π-interactions from the exact "
        "values above. One paragraph per interaction type present. "
        "Omit any interaction type not present in the data.\n\n"
        "### Semantic Events\n"
        f"Write one dedicated paragraph per event ({event_tags}). Each paragraph must state: "
        "event type, time window (ns), confidence score, severity, the exact evidence values, "
        "and the biological interpretation from the event data. "
        "If no events were detected, write one sentence stating this.\n\n"
        "## Discussion\n\n"
        "Write 2–3 focused paragraphs that:\n"
        "(a) Integrate the Results above into a mechanistic picture of binding-pose stability "
        "and interaction persistence, citing event tags where relevant.\n"
        "(b) Identify any tensions or notable patterns in the data "
        "(e.g., stable backbone but fluctuating ligand pose).\n"
        "(c) State what these findings suggest for further study, "
        "without overstating conclusions beyond the evidence.\n\n"
        "---\n\n"
        "STOP after Discussion. Do not add Conclusions, References, or any other section."
    )
    return MANUSCRIPT_SYSTEM_RULES + "\n\n" + context


def _build_chat_prompt(
    question: str,
    metrics: dict[str, Any],
    events: list[SemanticEvent],
    project_name: str,
) -> str:
    context = (
        f"PROJECT: {project_name}\n\n"
        f"=== TRAJECTORY METRICS ===\n{_metrics_to_text(metrics)}\n\n"
        f"=== DETECTED SEMANTIC EVENTS ===\n{_events_to_text(events)}\n\n"
        f"=== USER QUESTION ===\n{question}\n\n"
        "Answer the question using ONLY the metrics and events above. "
        "Cite event tags [E1], [E2], etc. when referencing specific events. "
        "If the evidence is insufficient to answer, say exactly what data is "
        "missing. Do NOT invent data or ask for column headers."
    )
    return SYSTEM_RULES + "\n\n" + context


# ---------------------------------------------------------------------------
# Ollama helpers
# ---------------------------------------------------------------------------

def ollama_generate(
    prompt: str,
    model: str = "llama3.1",
    base_url: str = "http://localhost:11434",
    timeout: int = 120,
) -> str:
    url = base_url.rstrip("/") + "/api/generate"
    r = requests.post(
        url, json={"model": model, "prompt": prompt, "stream": False}, timeout=timeout
    )
    r.raise_for_status()
    return r.json().get("response", "")


def ollama_generate_stream(
    prompt: str,
    model: str = "llama3.1",
    base_url: str = "http://localhost:11434",
) -> Generator[str, None, None]:
    url = base_url.rstrip("/") + "/api/generate"
    try:
        r = requests.post(
            url,
            json={"model": model, "prompt": prompt, "stream": True},
            stream=True,
            timeout=300,
        )
        r.raise_for_status()
        for line in r.iter_lines():
            if line:
                data = json.loads(line)
                if "response" in data:
                    yield data["response"]
    except Exception as exc:
        yield f"\n\n**Local LLM error:** `{exc}`\n"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_grounded_report(
    metrics: dict[str, Any],
    events: list[SemanticEvent],
    project_name: str,
    use_ollama: bool = True,
    model: str = "llama3.1",
    df: pd.DataFrame | None = None,
) -> tuple[str, str]:
    """
    Try to generate an LLM-grounded report via Ollama.
    If Ollama is unavailable or fails, fall back to the deterministic
    ``make_markdown_report`` which uses the real extracted data — never
    placeholder text.
    """
    try:
        prompt = build_grounded_prompt(metrics, events, project_name)
        return ollama_generate(prompt, model=model), "ollama"
    except Exception:
        if df is not None and not df.empty:
            report_df = df
        else:
            row: dict[str, Any] = {}
            t_max = 0.0
            for k, v in metrics.items():
                if k == "duration_ns":
                    t_max = float(v)
                elif isinstance(v, dict) and "mean" in v:
                    row[k] = v["mean"]
            row["time_ns"] = t_max
            report_df = pd.DataFrame([row])

        report = make_markdown_report(report_df, events, project_name)
        return report, "deterministic"


def generate_manuscript_stream(
    metrics: dict[str, Any],
    events: list[SemanticEvent],
    project_name: str,
    model: str = "llama3.1",
    base_url: str = "http://localhost:11434",
) -> Generator[str, None, None]:
    """
    Stream a full IMRaD-style scientific manuscript via Ollama.

    Uses the detailed ``build_manuscript_prompt`` which instructs the model to
    produce Abstract, Introduction, Methods, Results, Discussion, Conclusions,
    Limitations, and Data Availability — all strictly grounded in the supplied
    metrics and semantic events.

    Yields
    ------
    str
        Progressive text chunks from the Ollama streaming response.
    """
    prompt = build_manuscript_prompt(metrics, events, project_name)
    yield from ollama_generate_stream(prompt, model=model, base_url=base_url)


def answer_grounded_question_stream(
    question: str,
    metrics: dict[str, Any],
    events: list[SemanticEvent],
    project_name: str,
    use_ollama: bool = True,
    model: str = "llama3.1",
) -> Generator[str, None, None]:
    """Answer a user question only from extracted metrics/events."""
    question = (question or "").strip()
    if not question:
        yield "Ask a question about the extracted trajectory evidence, events, features, or report."
        return
    prompt = _build_chat_prompt(question, metrics, events, project_name)
    yield from ollama_generate_stream(prompt, model=model)
