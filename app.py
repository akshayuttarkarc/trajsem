from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from trajsem.demo import make_demo
from trajsem.events import detect_events
from trajsem.report import summarize_metrics
from trajsem.features.raw_md_research import FeatureConfig, mdanalysis_available, analyze_uploaded_raw_md
from trajsem.llm.grounded import (
    generate_grounded_report,
    generate_manuscript_stream,
    answer_grounded_question_stream,
)
from trajsem.pockets.adapters import normalize_pocket_volume_table, run_fpocket_on_pdb, tool_available, make_povme_template

st.set_page_config(page_title="TrajSem Research", page_icon="🧬", layout="wide", initial_sidebar_state="expanded")

CSS = """
<style>
.main .block-container { padding-top: 1.1rem; padding-bottom: 2rem; max-width: 1550px; }
.hero { border-radius: 28px; padding: 28px 32px; margin-bottom: 18px; background: linear-gradient(135deg, rgba(23, 84, 124, 0.14), rgba(15, 118, 110, 0.08)); border: 1px solid rgba(120, 144, 156, 0.22); }
.hero h1 { margin: 0; font-size: 2.45rem; letter-spacing: -0.04em; }
.hero p { font-size: 1.02rem; color: #52616b; max-width: 1100px; }
.event-card { border-radius: 18px; padding: 16px 18px; margin: 10px 0; border: 1px solid rgba(120,144,156,.22); background: rgba(250,252,255,.96); }
.event-title { font-weight: 760; font-size: 1.02rem; }
.small-muted { color: #697782; font-size: .91rem; }
.badge { border-radius: 999px; padding: 3px 9px; background: rgba(14, 116, 144, .10); font-size: .8rem; }
.warnbox { border-left: 4px solid #b7791f; padding: 12px 14px; background: rgba(251, 191, 36, .12); border-radius: 10px; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def clean_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    df.columns = [str(c).strip() for c in df.columns]
    if "time_ns" not in df.columns:
        df.insert(0, "time_ns", np.arange(len(df), dtype=float))
    return df


def merge_on_time(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    if left is None or left.empty:
        return right
    if right is None or right.empty:
        return left
    if "time_ns" in left.columns and "time_ns" in right.columns:
        return pd.merge_asof(left.sort_values("time_ns"), right.sort_values("time_ns"), on="time_ns", direction="nearest")
    return pd.concat([left.reset_index(drop=True), right.reset_index(drop=True)], axis=1)


# Columns that are index-like and should never be plotted as a descriptor
_INDEX_COLS = {"frame", "index", "id", "step", "snapshot", "timestep"}


def _is_index_col(name: str) -> bool:
    """Return True if a column looks like a row-index rather than a physical descriptor."""
    lower = name.strip().lower()
    return lower in _INDEX_COLS or lower.startswith("unnamed")


def line_fig(df: pd.DataFrame, cols: list[str], title: str):
    """Plot each descriptor in its own facet row so y-axes are independent."""
    cols = [c for c in cols if c in df.columns and not _is_index_col(c)]
    if not cols:
        return None
    plot_df = df[["time_ns"] + cols].melt(id_vars="time_ns", var_name="Metric", value_name="Value")
    fig = px.line(
        plot_df,
        x="time_ns",
        y="Value",
        color="Metric",
        facet_row="Metric",
        title=title,
        labels={"time_ns": "Time (ns)"},
    )
    # Give every facet row an independent y-axis so RMSD fluctuations are visible
    fig.update_yaxes(matches=None, showticklabels=True, title_text="")
    # Remove per-panel redundant "Metric=…" annotations and replace with the metric name
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    n = len(cols)
    row_h = max(180, min(300, 900 // n))  # adaptive height per row
    fig.update_layout(
        height=max(320, n * row_h),
        legend_title_text="Metric",
        margin=dict(l=10, r=10, t=55, b=10),
        title_font_size=15,
    )
    return fig


def event_dataframe(events):
    rows = []
    for i, e in enumerate(events, 1):
        rows.append({
            "id": f"E{i}",
            "event_type": e.event_type,
            "title": e.title,
            "start_ns": e.time_window_ns[0],
            "end_ns": e.time_window_ns[1],
            "confidence": e.confidence,
            "severity": e.severity,
            "entities": ", ".join([f"{k}: {v}" for k, v in e.entities.items()]) if isinstance(e.entities, dict) else str(e.entities),
            "evidence": ", ".join([f"{k}: {v}" for k, v in e.evidence.items()]) if isinstance(e.evidence, dict) else str(e.evidence),
        })
    return pd.DataFrame(rows)


with st.sidebar:
    st.title("TrajSem Research")
    st.caption("Local, evidence-grounded MD trajectory semantics")
    mode = st.radio("Input mode", ["No-input demo", "Descriptor CSV", "Raw MD trajectory", "Hybrid: raw/CSV + pocket table"], help="Demo runs immediately. Raw MD requires MDAnalysis. Pocket tables can come from fpocket, MDpocket, POVME, or your own CSV.")
    project_name = st.text_input("Project name", value="TrajSem research analysis")
    st.divider()
    st.subheader("Feature settings")
    stride = st.number_input("Raw trajectory stride", min_value=1, max_value=10000, value=10, step=1)
    ligand_sel = st.text_input("Ligand selection", value="not protein and not resname SOL HOH WAT TIP3 NA CL K MG CA")
    water_sel = st.text_input("Water selection", value="resname SOL HOH WAT TIP3 TIP4 TIP5")
    ligand_pi_sel = st.text_input("Ligand aromatic ring selection (optional)", value="", help="For publication-grade π interactions, provide explicit ligand ring atoms, e.g. resname LIG and name C1 C2 C3 C4 C5 C6")
    contact_cutoff = st.number_input("Residue contact cutoff (Å)", min_value=2.0, max_value=10.0, value=4.5, step=0.1)
    water_cutoff = st.number_input("Water-bridge cutoff (Å)", min_value=2.5, max_value=4.5, value=3.5, step=0.1)
    pi_cutoff = st.number_input("π centroid cutoff (Å)", min_value=3.5, max_value=7.5, value=5.5, step=0.1)
    st.divider()
    st.subheader("LLM settings")
    ollama_model = st.text_input("Ollama model", value="llama3.1",
        help="Any model pulled via 'ollama pull <name>', e.g. llama3.1, mistral, gemma3")
    ollama_url = st.text_input("Ollama base URL", value="http://localhost:11434")
    use_ollama = True

st.markdown("""
<div class="hero">
  <h1>TrajSem Research</h1>
  <p>A local research-grade scaffold for semantic extraction from MD trajectories: raw trajectory descriptors, residue-level interaction geometry, pocket-volume integration, grounded report generation, hallucination verification, and expert annotation evaluation.</p>
</div>
""", unsafe_allow_html=True)

st.info("This app runs locally. External tools such as fpocket, MDpocket, POVME, and Ollama are optional. When unavailable, the app still accepts their CSV outputs or uses deterministic report generation.")

base_df: pd.DataFrame | None = None
meta = {}

if mode == "No-input demo":
    base_df = make_demo()
    meta = {"source": "synthetic_demo", "note": "Demonstration descriptors with known events."}

elif mode == "Descriptor CSV":
    desc = st.file_uploader("Upload descriptor CSV", type=["csv"])
    if desc:
        base_df = clean_csv(desc)
        meta = {"source": desc.name}
    else:
        st.warning("Upload a descriptor CSV or switch to No-input demo.")

elif mode == "Raw MD trajectory":
    if not mdanalysis_available():
        st.error("MDAnalysis is not installed. Run: python -m pip install -r requirements-md.txt")
    top = st.file_uploader("Topology / structure file", type=["pdb", "gro", "psf", "prmtop", "mol2", "pqr", "tpr"])
    traj = st.file_uploader("Trajectory file", type=["xtc", "trr", "dcd", "nc", "netcdf", "pdb"])
    if top and traj and mdanalysis_available():
        with st.spinner("Extracting raw trajectory descriptors locally..."):
            cfg = FeatureConfig(ligand_selection=ligand_sel, water_selection=water_sel, ligand_pi_selection=ligand_pi_sel, stride=int(stride), contact_cutoff_A=float(contact_cutoff), water_bridge_cutoff_A=float(water_cutoff), pi_centroid_cutoff_A=float(pi_cutoff))
            base_df, meta = analyze_uploaded_raw_md(top.getvalue(), top.name, traj.getvalue(), traj.name, cfg)
    elif not (top and traj):
        st.warning("Upload both a topology/structure file and a trajectory file.")

else:
    left, right = st.columns(2)
    with left:
        st.subheader("Base descriptors")
        desc = st.file_uploader("Descriptor CSV", type=["csv"], key="hybrid_desc")
        if desc:
            base_df = clean_csv(desc)
        else:
            st.caption("No descriptor CSV uploaded.")
    with right:
        st.subheader("Pocket-volume table")
        pocket = st.file_uploader("fpocket/MDpocket/POVME/custom CSV", type=["csv"], key="pocket_csv")
        if pocket:
            pocket_df = normalize_pocket_volume_table(pocket)
            base_df = merge_on_time(base_df if base_df is not None else pd.DataFrame(), pocket_df)
            meta["pocket_table"] = pocket.name

    st.subheader("Optional single-frame fpocket check")
    pdb = st.file_uploader("Representative PDB frame for fpocket", type=["pdb"], key="fpocket_pdb")
    if pdb:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / pdb.name
            p.write_bytes(pdb.getvalue())
            result = run_fpocket_on_pdb(p)
            if result.available:
                st.success(result.message)
                if not result.table.empty:
                    st.dataframe(result.table, use_container_width=True)
            else:
                st.warning(result.message)

if base_df is None or base_df.empty:
    st.stop()

# Normalize numeric columns.
for col in base_df.columns:
    if col != "time_ns":
        try:
            base_df[col] = pd.to_numeric(base_df[col])
        except (ValueError, TypeError):
            pass

metrics = summarize_metrics(base_df)
events = detect_events(base_df)
report, report_mode = generate_grounded_report(metrics, events, project_name, use_ollama=use_ollama, model=ollama_model, df=base_df)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Frames", f"{len(base_df):,}")
k2.metric("Duration", f"{metrics.get('duration_ns', 0)} ns")
k3.metric("Events", len(events))
k4.metric("Report mode", report_mode)

tabs = st.tabs(["Overview", "Features", "Semantic events", "Grounded report", "LLM chat", "Exports"])

with tabs[0]:
    st.subheader("Trajectory descriptor overview")
    cols = [c for c in ["protein_rmsd_A", "ligand_rmsd_A", "radius_of_gyration_A", "pocket_volume_A3", "hbond_approx_count", "salt_bridge_count", "water_bridge_count", "pi_interaction_count"] if c in base_df.columns]
    fig = line_fig(base_df, cols, "Core trajectory descriptors")
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("**Metadata:**")
    for k, v in meta.items():
        st.write(f"- **{k}**: {v}")

with tabs[1]:
    st.subheader("Extracted features")
    st.dataframe(base_df, use_container_width=True, height=420)
    # Exclude index-like columns (frame, step, index, …) — they are not physical descriptors
    numeric_cols = [
        c for c in base_df.select_dtypes(include="number").columns
        if c != "time_ns" and not _is_index_col(c)
    ]
    selected = st.multiselect("Plot custom descriptors", numeric_cols, default=numeric_cols[: min(4, len(numeric_cols))])
    fig = line_fig(base_df, selected, "Selected descriptors")
    if fig:
        st.plotly_chart(fig, use_container_width=True)

with tabs[2]:
    st.subheader("Evidence-linked semantic events")
    if not events:
        st.write("No semantic events passed the current thresholds.")
    for i, ev in enumerate(events, 1):
        st.markdown(f"""
<div class="event-card">
  <div class="event-title">E{i}. {ev.title}</div>
  <div class="small-muted">{ev.event_type} · {ev.time_window_ns[0]:.3f}–{ev.time_window_ns[1]:.3f} ns · confidence {ev.confidence:.2f} · severity {ev.severity}</div>
  <p>{ev.interpretation}</p>
  <span class="badge">Evidence linked</span>
</div>
""", unsafe_allow_html=True)
    edf = event_dataframe(events)
    if not edf.empty:
        st.dataframe(edf, use_container_width=True)

with tabs[3]:
    st.subheader("Evidence-grounded report")
    st.download_button("⬇ Download report.md", report.encode("utf-8"), file_name="trajsem_report.md", mime="text/markdown")
    st.markdown(report)

    st.divider()
    st.subheader("📄 Results & Discussion — LLM Scientific Report")
    st.caption(
        "Generates **Results** and **Discussion** sections grounded strictly in the "
        "extracted descriptor statistics and detected semantic events. "
        f"Uses Ollama model **{ollama_model}**. Only data present in this session is reported."
    )

    if "manuscript_text" not in st.session_state:
        st.session_state["manuscript_text"] = ""

    if st.button("✍️ Generate Results & Discussion", type="primary", key="gen_manuscript"):
        st.session_state["manuscript_text"] = ""
        manuscript_placeholder = st.empty()
        status = st.status("🧬 Writing manuscript — this may take 1–3 minutes depending on your model...", expanded=True)
        try:
            collected: list[str] = []
            stream = generate_manuscript_stream(
                metrics, events, project_name,
                model=ollama_model,
                base_url=ollama_url,
            )
            for chunk in stream:
                collected.append(chunk)
                manuscript_placeholder.markdown("".join(collected))
            st.session_state["manuscript_text"] = "".join(collected)
            status.update(label="✅ Manuscript complete!", state="complete", expanded=False)
        except Exception as exc:
            status.update(label="❌ Generation failed", state="error", expanded=True)
            st.error(f"Ollama error: {exc}")

    elif st.session_state.get("manuscript_text"):
        st.markdown(st.session_state["manuscript_text"])

    if st.session_state.get("manuscript_text"):
        st.download_button(
            "⬇ Download manuscript.md",
            st.session_state["manuscript_text"].encode("utf-8"),
            file_name="trajsem_manuscript.md",
            mime="text/markdown",
            key="dl_manuscript",
        )

with tabs[4]:
    st.subheader("LLM chat over extracted evidence")
    st.caption("Ask about extracted features, interaction events, pocket validation, report claims, or missing evidence. Answers are constrained to the descriptor table and detected events.")
    default_q = "What features were extracted and which events are publication-ready?"
    q = st.text_area("Question", value=default_q, height=90)
    if st.button("Ask TrajSem", type="primary"):
        with st.spinner("Thinking..."):
            ans_stream = answer_grounded_question_stream(q, metrics, events, project_name, use_ollama=use_ollama, model=ollama_model)
            st.write_stream(ans_stream)

with tabs[5]:
    st.subheader("Export research artifacts")
    st.download_button("Download descriptors.csv", base_df.to_csv(index=False).encode("utf-8"), file_name="trajsem_descriptors.csv", mime="text/csv")
    st.download_button("Download events.json", json.dumps([e.to_dict() for e in events], indent=2).encode("utf-8"), file_name="trajsem_events.json", mime="application/json")
    with tempfile.TemporaryDirectory() as td:
        povme_path = Path(td) / "trajsem_povme_template.cfg"
        make_povme_template(str(povme_path))
        st.download_button("Download POVME template", povme_path.read_bytes(), file_name="trajsem_povme_template.cfg", mime="text/plain")
