# TrajSem Research

TrajSem Research is a local, cross-platform application for semantic extraction from molecular dynamics trajectories. It is designed as a research scaffold for a paper-grade system, not merely a visual demo.

It runs locally on Windows, macOS, and Linux through a browser-based desktop UI.

## Fast start

```bash
python -m pip install -r requirements.txt
python run_app.py
```

Open the Streamlit URL shown in the terminal and choose **No-input demo**.

For full step-by-step instructions, see `INSTALL_AND_FEATURES.md`.

## Raw MD trajectory mode

```bash
python -m pip install -r requirements-md.txt
python run_app.py
```

Then upload a topology/structure file and a trajectory file. The app uses MDAnalysis when installed.

## Optional research extras

These are external tools, not bundled Python dependencies:

- `fpocket` on PATH for single-frame pocket detection.
- `MDpocket`/fpocket suite for trajectory pocket analysis.
- `POVME` for pocket-volume calculations.
- `Ollama` running locally for local LLM report generation and evidence-grounded chat.

If these tools are not installed, TrajSem still accepts their CSV outputs. This is intentional: it avoids forcing users to solve system-level dependencies before using the app.

## What is implemented

- Local Streamlit desktop-style UI.
- No-input demo mode.
- Descriptor CSV mode.
- Raw trajectory feature extraction using MDAnalysis.
- Residue-level ligand contact distances and contact-presence signals.
- Approximate polar-contact/H-bond screening.
- Explicit water-bridge detection through bridging-water geometry.
- π-interaction detection using aromatic centroid and plane-angle criteria.
- Salt-bridge count screening.
- Pocket-volume table ingestion from fpocket, MDpocket, POVME, or custom CSV.
- Stronger cryptic-pocket validation using pocket volume plus structural support descriptors.
- Optional single-frame fpocket adapter.
- Evidence-linked semantic event extraction.
- Local Ollama report generation and chat with deterministic fallback.
- Hallucination/grounding verifier.
- Expert annotation upload with interval IoU, precision, recall, and F1.
- Export of descriptors, event JSON, report Markdown, verification JSON, and POVME template.

## Minimal commands

Base UI:

```bash
python -m pip install -r requirements.txt
python run_app.py
```

Raw trajectory analysis:

```bash
python -m pip install -r requirements-md.txt
python run_app.py
```

## Expert annotation format

Upload JSON or JSONL like this:

```json
[
  {
    "event_type": "ligand_pose_drift",
    "time_window_ns": [55.0, 82.0],
    "entities": {"ligand": "LIG"},
    "notes": "Expert rationale."
  }
]
```
