from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable
import re
import numpy as np
import pandas as pd

from .schemas import SemanticEvent, EvidenceRef

# Backward-compatible name used by older report code.
Event = SemanticEvent


def _segments(mask: np.ndarray, min_len: int = 5) -> list[tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool)
    if len(mask) == 0:
        return []
    edges = np.diff(mask.astype(int), prepend=0, append=0)
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0] - 1
    return [(int(s), int(e)) for s, e in zip(starts, ends) if e - s + 1 >= min_len]


def _largest_segment(mask: np.ndarray, min_len: int = 5) -> tuple[int, int] | None:
    segs = _segments(mask, min_len=min_len)
    if not segs:
        return None
    return max(segs, key=lambda p: p[1] - p[0])


def _required(df: pd.DataFrame, columns: Iterable[str]) -> bool:
    return all(c in df.columns for c in columns)


def _time(df: pd.DataFrame) -> np.ndarray:
    if "time_ns" in df.columns:
        return df["time_ns"].to_numpy(float)
    return np.arange(len(df), dtype=float)


def _frame_window(df: pd.DataFrame, s: int, e: int) -> tuple[int, int] | None:
    if "frame" in df.columns:
        return int(df["frame"].iloc[s]), int(df["frame"].iloc[e])
    return None


def _event(event_type: str, title: str, df: pd.DataFrame, s: int, e: int, metric: str, evidence: dict, interpretation: str, confidence: float, severity: str, entities: dict | None = None) -> SemanticEvent:
    t = _time(df)
    fw = _frame_window(df, s, e)
    refs = [EvidenceRef(source="descriptor_table", metric=metric, value=evidence, frame=fw[0] if fw else None, time_ns=float(t[s]), description=title)]
    return SemanticEvent(
        event_type=event_type,
        title=title,
        time_window_ns=(float(t[s]), float(t[e])),
        frame_window=fw,
        entities=entities or {"metric": metric},
        evidence=evidence,
        evidence_refs=refs,
        interpretation=interpretation,
        confidence=round(float(np.clip(confidence, 0.0, 0.99)), 3),
        severity=severity,
        verification_status="evidence_linked",
    )


def _baseline(x: np.ndarray, n: int) -> float:
    if len(x) == 0:
        return float("nan")
    end = min(len(x), max(10, n // 5))
    return float(np.nanmedian(x[:end]))


def detect_events(df: pd.DataFrame) -> list[SemanticEvent]:
    df = df.copy()
    if "time_ns" not in df.columns:
        df.insert(0, "time_ns", np.arange(len(df), dtype=float))
    n = len(df)
    min_len = max(5, int(0.04 * max(n, 1)))
    events: list[SemanticEvent] = []

    if _required(df, ["ligand_rmsd_A"]):
        x = df["ligand_rmsd_A"].to_numpy(float)
        early = _baseline(x, n)
        high = x > early + 1.25
        seg = _largest_segment(high, min_len=min_len)
        if seg:
            s, e = seg
            evidence = {"early_median_A": round(early, 3), "window_mean_A": round(float(np.nanmean(x[s:e+1])), 3), "window_max_A": round(float(np.nanmax(x[s:e+1])), 3)}
            events.append(_event("ligand_pose_drift", "Ligand pose drift detected", df, s, e, "ligand_rmsd_A", evidence, "The ligand deviates substantially from the initial pose during this interval. This supports possible binding-pose instability or a transition to a secondary pose.", (np.nanmean(x[s:e+1]) - early) / 3.5, "high"))

    if _required(df, ["pocket_volume_A3"]):
        x = df["pocket_volume_A3"].to_numpy(float)
        base = _baseline(x, n)
        threshold = base + max(80.0, 1.75 * float(np.nanstd(x[: max(10, n // 4)])))
        seg = _largest_segment(x > threshold, min_len=min_len)
        if seg:
            s, e = seg
            support = []
            support_values = {}
            if "binding_site_sasa_A2" in df.columns:
                y = df["binding_site_sasa_A2"].to_numpy(float)
                ybase = _baseline(y, n)
                ydelta = float(np.nanmean(y[s:e+1]) - ybase)
                support_values["binding_site_sasa_delta_A2"] = round(ydelta, 2)
                if ydelta > max(25.0, 0.05 * abs(ybase)):
                    support.append("binding-site SASA increase")
            for c in df.columns:
                if "loop" in c.lower() and c.endswith("_A"):
                    y = df[c].to_numpy(float)
                    ybase = _baseline(y, n)
                    ydelta = float(np.nanmean(y[s:e+1]) - ybase)
                    support_values[f"{c}_delta"] = round(ydelta, 3)
                    if ydelta > 0.75:
                        support.append(f"{c} increase")
            if "pocket_lining_residue_count" in df.columns:
                y = df["pocket_lining_residue_count"].to_numpy(float)
                support_values["mean_pocket_lining_residue_count"] = round(float(np.nanmean(y[s:e+1])), 2)
                if np.nanmean(y[s:e+1]) >= 4:
                    support.append("multiple lining residues reported")
            validation = "validated_candidate" if len(support) >= 1 else "volume_only_candidate"
            evidence = {"baseline_A3": round(base, 2), "window_mean_A3": round(float(np.nanmean(x[s:e+1])), 2), "window_max_A3": round(float(np.nanmax(x[s:e+1])), 2), "threshold_A3": round(threshold, 2), "structural_support": support or ["none supplied"], "cryptic_pocket_validation": validation, **support_values}
            interpretation = "Pocket volume expands above baseline and at least one structural support descriptor changes in the same interval. This supports a cryptic-pocket candidate that should be checked against representative frames and replicate simulations." if validation == "validated_candidate" else "Pocket volume expands above baseline, but no independent structural-support descriptor was supplied. Treat this as a volume-only pocket-opening candidate, not a publication-grade cryptic-pocket claim."
            events.append(_event("transient_pocket_opening", "Transient pocket-opening candidate", df, s, e, "pocket_volume_A3", evidence, interpretation, (np.nanmax(x[s:e+1]) - base) / max(250, base), "moderate", {"metric": "pocket_volume_A3", "validation": validation}))

    # Explicit occupancy columns, e.g. hbond_ASP147_occupancy or contact_PHE91_occupancy.
    for col in list(df.columns):
        low = col.lower()
        if low.startswith("hbond_") and ("occupancy" in low or "present" in low):
            x = df[col].to_numpy(float)
            seg = _largest_segment(x > 0.55, min_len=min_len)
            if seg:
                s, e = seg
                entity = re.sub(r"^hbond_", "", low).replace("_occupancy", "").replace("_present", "").upper()
                evidence = {"mean_occupancy": round(float(np.nanmean(x[s:e+1])), 3), "max_occupancy": round(float(np.nanmax(x[s:e+1])), 3)}
                events.append(_event("persistent_hydrogen_bond", f"Persistent hydrogen-bond signal: {entity}", df, s, e, col, evidence, "The hydrogen-bond signal persists above the occupancy threshold. Confirm donor/acceptor geometry before making a mechanistic claim.", float(np.nanmean(x[s:e+1])), "moderate", {"interaction": entity}))
        elif low.startswith("contact_") and low.endswith("_present"):
            x = df[col].to_numpy(float)
            early = float(np.nanmean(x[: max(10, n // 5)]))
            late_loss = x < 0.35
            seg = _largest_segment(late_loss, min_len=min_len)
            if seg and early > 0.55:
                s, e = seg
                entity = col[len("contact_"):-len("_present")].upper()
                evidence = {"early_mean_presence": round(early, 3), "window_mean_presence": round(float(np.nanmean(x[s:e+1])), 3)}
                events.append(_event("residue_contact_loss", f"Residue contact loss: {entity}", df, s, e, col, evidence, "A residue-ligand contact present early in the simulation is lost for a sustained interval.", early - float(np.nanmean(x[s:e+1])), "moderate", {"residue": entity}))
        elif low.startswith("contact_") and low.endswith("_distance_a"):
            x = df[col].to_numpy(float)
            early = _baseline(x, n)
            seg = _largest_segment(x > early + 2.0, min_len=min_len)
            if seg and early < 5.0:
                s, e = seg
                entity = low[len("contact_"):-len("_distance_a")].upper()
                evidence = {"early_median_distance_A": round(early, 3), "window_mean_distance_A": round(float(np.nanmean(x[s:e+1])), 3)}
                events.append(_event("residue_ligand_separation", f"Residue-ligand separation: {entity}", df, s, e, col, evidence, "The residue-ligand minimum distance increases substantially from its early baseline.", (float(np.nanmean(x[s:e+1])) - early) / 4, "moderate", {"residue": entity}))
        elif low.startswith("contact_") and low.endswith("_occupancy"):
            x = df[col].to_numpy(float)
            early = float(np.nanmean(x[: max(10, n // 5)]))
            late_loss = x < 0.35
            seg = _largest_segment(late_loss, min_len=min_len)
            if seg and early > 0.55:
                s, e = seg
                entity = low[len("contact_"):-len("_occupancy")].upper()
                evidence = {"early_mean_occupancy": round(early, 3), "window_mean_occupancy": round(float(np.nanmean(x[s:e+1])), 3)}
                events.append(_event("residue_contact_loss", f"Residue contact loss: {entity}", df, s, e, col, evidence, "A residue-ligand contact present early in the simulation is lost for a sustained interval.", early - float(np.nanmean(x[s:e+1])), "moderate", {"residue": entity}))


    if "water_bridge_count" in df.columns:
        x = df["water_bridge_count"].to_numpy(float)
        seg = _largest_segment(x >= 1, min_len=min_len)
        if seg:
            s, e = seg
            residues = ""
            if "water_bridge_residues" in df.columns:
                vals = [str(v) for v in df["water_bridge_residues"].iloc[s:e+1].dropna().tolist() if str(v)]
                residues = ";".join(sorted(set(";".join(vals).split(";"))))[:250] if vals else ""
            evidence = {"window_mean_bridge_count": round(float(np.nanmean(x[s:e+1])), 3), "window_max_bridge_count": int(np.nanmax(x[s:e+1])), "residues": residues, "definition": "water oxygen simultaneously within cutoff of ligand polar atom and protein polar atom"}
            events.append(_event("water_bridge_network", "Persistent water-bridge network", df, s, e, "water_bridge_count", evidence, "Explicit water molecules mediate ligand-protein polar connectivity during this interval. This is a structural water-bridge signal, not a binding free-energy estimate.", min(0.95, float(np.nanmean(x[s:e+1])) / 3.0 + 0.25), "moderate", {"interaction": "ligand-water-protein"}))

    if "pi_interaction_count" in df.columns:
        x = df["pi_interaction_count"].to_numpy(float)
        seg = _largest_segment(x >= 1, min_len=min_len)
        if seg:
            s, e = seg
            residues = ""
            if "pi_interaction_residues" in df.columns:
                vals = [str(v) for v in df["pi_interaction_residues"].iloc[s:e+1].dropna().tolist() if str(v)]
                residues = ";".join(sorted(set(";".join(vals).split(";"))))[:250] if vals else ""
            mode = str(df["pi_validation_mode"].iloc[s]) if "pi_validation_mode" in df.columns else "unknown"
            evidence = {"window_mean_pi_count": round(float(np.nanmean(x[s:e+1])), 3), "window_max_pi_count": int(np.nanmax(x[s:e+1])), "residues": residues, "validation_mode": mode}
            if "pi_min_centroid_distance_A" in df.columns:
                evidence["min_centroid_distance_A"] = round(float(np.nanmin(df["pi_min_centroid_distance_A"].to_numpy(float)[s:e+1])), 3)
            if "pi_min_plane_angle_deg" in df.columns:
                evidence["min_plane_angle_deg"] = round(float(np.nanmin(df["pi_min_plane_angle_deg"].to_numpy(float)[s:e+1])), 2)
            interp = "A protein aromatic ring and ligand ring satisfy centroid-distance and plane-angle criteria. This supports a π-interaction assignment for the interval." if mode in {"user_selection", "planar_ligand_fallback"} else "A π-interaction count was supplied, but ligand ring validation mode is weak or unknown; inspect atom selections before publication."
            conf = 0.82 if mode == "user_selection" else 0.62 if mode == "planar_ligand_fallback" else 0.45
            events.append(_event("pi_interaction", "π-interaction candidate", df, s, e, "pi_interaction_count", evidence, interp, conf, "moderate", {"interaction": "aromatic_pi", "residues": residues}))

    if "hbond_approx_count" in df.columns:
        x = df["hbond_approx_count"].to_numpy(float)
        base = _baseline(x, n)
        seg = _largest_segment(x > base + 2, min_len=min_len)
        if seg:
            s, e = seg
            evidence = {"baseline_count": round(base, 3), "window_mean_count": round(float(np.nanmean(x[s:e+1])), 3), "note": "Approximate N/O/S close-contact count"}
            events.append(_event("hydrogen_bond_network_increase", "Approximate hydrogen-bond network increase", df, s, e, "hbond_approx_count", evidence, "Close polar contacts increase relative to baseline. This should be treated as a screening signal until donor-acceptor geometry is validated.", (float(np.nanmean(x[s:e+1])) - base) / 5, "low"))

    for col in ["loop_185_193_displacement_A", "binding_site_sasa_A2"]:
        if col in df.columns:
            x = df[col].to_numpy(float)
            base = _baseline(x, n)
            seg = _largest_segment(x > base + max(1.0, 1.5 * float(np.nanstd(x))), min_len=min_len)
            if seg:
                s, e = seg
                evidence = {"baseline": round(base, 3), "window_mean": round(float(np.nanmean(x[s:e+1])), 3), "window_max": round(float(np.nanmax(x[s:e+1])), 3)}
                title = "Loop displacement event" if "loop" in col else "Binding-site exposure increase"
                events.append(_event("conformational_shift", title, df, s, e, col, evidence, "The descriptor shows a sustained displacement/exposure increase. Structural frame inspection is required to assign a specific conformational mechanism.", (float(np.nanmean(x[s:e+1])) - base) / max(3, abs(base)), "moderate"))

    return sorted(events, key=lambda ev: (ev.time_window_ns[0], ev.event_type))
