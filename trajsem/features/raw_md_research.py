from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import Iterable
import math
import tempfile

import numpy as np
import pandas as pd


@dataclass
class FeatureConfig:
    protein_selection: str = "protein"
    ligand_selection: str = "not protein and not resname SOL HOH WAT TIP3 TIP4 TIP5 NA CL K MG CA"
    water_selection: str = "resname SOL HOH WAT TIP3 TIP4 TIP5"
    ligand_pi_selection: str = ""  # Optional: e.g. "resname LIG and name C1 C2 C3 C4 C5 C6"
    binding_site_cutoff_A: float = 5.0
    contact_cutoff_A: float = 4.5
    hbond_distance_A: float = 3.5
    salt_bridge_cutoff_A: float = 4.0
    water_bridge_cutoff_A: float = 3.5
    pi_centroid_cutoff_A: float = 5.5
    pi_parallel_angle_max_deg: float = 35.0
    stride: int = 10
    max_residue_contacts: int = 80


def mdanalysis_available() -> bool:
    try:
        import MDAnalysis  # noqa: F401
        return True
    except Exception:
        return False


def _safe_name(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in s)


def _res_label(residue) -> str:
    return f"{residue.resname}{residue.resid}"


def _rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    if len(P) == 0 or len(Q) == 0 or P.shape != Q.shape:
        return float("nan")
    from MDAnalysis.analysis.rms import rmsd as mda_rmsd
    return float(mda_rmsd(P, Q, center=True, superposition=True))


def _min_distance(a, b) -> float:
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    from MDAnalysis.lib.distances import distance_array
    return float(np.min(distance_array(a.positions, b.positions)))


def _residue_contact_summary(ligand, residues, cutoff: float, max_residues: int) -> dict[str, float]:
    if len(ligand) == 0:
        return {}
    out: dict[str, float] = {}
    for res in residues[:max_residues]:
        atoms = res.atoms.select_atoms("not name H* and not type H*")
        if len(atoms) == 0:
            atoms = res.atoms
        d = _min_distance(atoms, ligand)
        label = _res_label(res)
        out[f"contact_{label}_distance_A"] = round(d, 3) if not math.isnan(d) else np.nan
        out[f"contact_{label}_present"] = int(d <= cutoff) if not math.isnan(d) else 0
    return out


def _binding_site_residues(universe, ligand, cutoff: float):
    if len(ligand) == 0:
        return []
    protein = universe.select_atoms("protein")
    if len(protein) == 0:
        return []
    from MDAnalysis.lib.distances import distance_array
    d = distance_array(protein.positions, ligand.positions)
    atom_indices = np.where(d.min(axis=1) <= cutoff)[0]
    residues = protein[atom_indices].residues
    return list(residues)


def _charged_groups(universe):
    positive = universe.select_atoms("protein and (resname ARG LYS HIS HIE HID HIP) and name NZ NH1 NH2 NE NE2 ND1")
    negative = universe.select_atoms("protein and (resname ASP GLU) and name OD1 OD2 OE1 OE2")
    return positive, negative


def _salt_bridge_count(positive, negative, cutoff: float) -> int:
    if len(positive) == 0 or len(negative) == 0:
        return 0
    from MDAnalysis.lib.distances import distance_array
    return int((distance_array(positive.positions, negative.positions) <= cutoff).sum())


def _hbond_approx_count(universe, ligand, cutoff: float) -> int:
    if len(ligand) == 0:
        return 0
    prot_heavy = universe.select_atoms("protein and (name O* N* S*)")
    lig_heavy = ligand.select_atoms("name O* N* S*")
    if len(prot_heavy) == 0 or len(lig_heavy) == 0:
        return 0
    from MDAnalysis.lib.distances import distance_array
    return int((distance_array(prot_heavy.positions, lig_heavy.positions) <= cutoff).sum())


def _water_bridge_summary(universe, ligand, water_selection: str, cutoff: float) -> dict[str, object]:
    """Detect ligand-water-protein bridges by explicit bridging-water geometry.

    Definition used here: a water oxygen is counted as bridging when it is within
    cutoff of at least one ligand polar atom and at least one protein polar atom.
    This is stronger than a generic close-contact count, but still not a full
    energetic or lifetime analysis. Residue labels indicate protein residues
    connected through at least one bridging water in the sampled frame.
    """
    out = {"water_bridge_count": 0, "water_bridge_residue_count": 0, "water_bridge_residues": ""}
    if len(ligand) == 0:
        return out
    waters = universe.select_atoms(f"({water_selection}) and (name O* OW OH2)")
    lig_polar = ligand.select_atoms("name O* N* S*")
    prot_polar = universe.select_atoms("protein and (name O* N* S*)")
    if len(waters) == 0 or len(lig_polar) == 0 or len(prot_polar) == 0:
        return out
    from MDAnalysis.lib.distances import distance_array
    dwl = distance_array(waters.positions, lig_polar.positions).min(axis=1)
    dwp = distance_array(waters.positions, prot_polar.positions)
    bridge_water_idx = np.where((dwl <= cutoff) & (dwp.min(axis=1) <= cutoff))[0]
    if len(bridge_water_idx) == 0:
        return out
    residues = set()
    for wi in bridge_water_idx:
        near_atom_idx = np.where(dwp[wi] <= cutoff)[0]
        for atom in prot_polar[near_atom_idx]:
            residues.add(_res_label(atom.residue))
    labels = sorted(residues)
    return {
        "water_bridge_count": int(len(bridge_water_idx)),
        "water_bridge_residue_count": int(len(labels)),
        "water_bridge_residues": ";".join(labels[:20]),
    }


_PROTEIN_RING_ATOMS = {
    "PHE": ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    "TYR": ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    "TRP": ["CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2"],
    "HIS": ["CG", "ND1", "CD2", "CE1", "NE2"],
    "HID": ["CG", "ND1", "CD2", "CE1", "NE2"],
    "HIE": ["CG", "ND1", "CD2", "CE1", "NE2"],
    "HIP": ["CG", "ND1", "CD2", "CE1", "NE2"],
}


def _plane_normal(coords: np.ndarray) -> np.ndarray | None:
    if coords.shape[0] < 3:
        return None
    centered = coords - coords.mean(axis=0)
    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
    except Exception:
        return None
    n = vh[-1]
    norm = np.linalg.norm(n)
    if norm == 0:
        return None
    return n / norm


def _protein_aromatic_rings(universe):
    rings = []
    for res in universe.select_atoms("protein and resname PHE TYR TRP HIS HID HIE HIP").residues:
        names = _PROTEIN_RING_ATOMS.get(res.resname.upper())
        if not names:
            continue
        atoms = res.atoms.select_atoms("name " + " ".join(names))
        if len(atoms) >= 5:
            rings.append((_res_label(res), atoms))
    return rings


def _ligand_pi_atoms(ligand, ligand_pi_selection: str, universe=None):
    if len(ligand) == 0:
        return None, "none"
    if ligand_pi_selection.strip() and universe is not None:
        ag = universe.select_atoms(ligand_pi_selection)
        if len(ag) >= 5:
            return ag, "user_selection"
    # Conservative fallback: only use a whole ligand pseudo-plane when it is small and planar-ish.
    heavy = ligand.select_atoms("not name H* and not type H*")
    if 5 <= len(heavy) <= 12:
        n = _plane_normal(heavy.positions)
        if n is not None:
            residual = np.abs((heavy.positions - heavy.positions.mean(axis=0)) @ n)
            if float(np.mean(residual)) <= 0.35:
                return heavy, "planar_ligand_fallback"
    return None, "no_valid_ligand_ring"


def _pi_interaction_summary(universe, ligand, ligand_pi_selection: str, cutoff: float, parallel_max_deg: float) -> dict[str, object]:
    """Detect protein-aromatic to ligand-ring pi contacts with centroid and plane geometry."""
    out = {
        "pi_interaction_count": 0,
        "pi_interaction_residues": "",
        "pi_validation_mode": "no_valid_ligand_ring",
        "pi_min_centroid_distance_A": np.nan,
        "pi_min_plane_angle_deg": np.nan,
    }
    lig_ring, mode = _ligand_pi_atoms(ligand, ligand_pi_selection, universe=universe)
    out["pi_validation_mode"] = mode
    if lig_ring is None or len(lig_ring) < 5:
        return out
    lig_centroid = lig_ring.positions.mean(axis=0)
    lig_normal = _plane_normal(lig_ring.positions)
    if lig_normal is None:
        return out
    hits = []
    min_d = np.inf
    min_angle = np.inf
    for label, atoms in _protein_aromatic_rings(universe):
        prot_centroid = atoms.positions.mean(axis=0)
        prot_normal = _plane_normal(atoms.positions)
        if prot_normal is None:
            continue
        d = float(np.linalg.norm(prot_centroid - lig_centroid))
        cosang = float(np.clip(abs(np.dot(prot_normal, lig_normal)), 0.0, 1.0))
        angle = float(np.degrees(np.arccos(cosang)))  # 0 parallel; 90 perpendicular
        min_d = min(min_d, d)
        min_angle = min(min_angle, angle)
        is_parallel_stack = d <= cutoff and angle <= parallel_max_deg
        is_t_shape = d <= cutoff and 55.0 <= angle <= 90.0
        if is_parallel_stack or is_t_shape:
            hits.append(label)
    if np.isfinite(min_d):
        out["pi_min_centroid_distance_A"] = round(min_d, 3)
    if np.isfinite(min_angle):
        out["pi_min_plane_angle_deg"] = round(min_angle, 2)
    labels = sorted(set(hits))
    out["pi_interaction_count"] = int(len(labels))
    out["pi_interaction_residues"] = ";".join(labels[:20])
    return out


def extract_raw_md_features(
    topology_path: str | Path,
    trajectory_path: str | Path,
    config: FeatureConfig | None = None,
    export_frames_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    import MDAnalysis as mda

    cfg = config or FeatureConfig()
    u = mda.Universe(str(topology_path), str(trajectory_path))
    protein = u.select_atoms(cfg.protein_selection)
    ligand = u.select_atoms(cfg.ligand_selection)
    if len(protein) == 0:
        protein = u.atoms

    u.trajectory[0]
    ref_protein = protein.positions.copy()
    ref_ligand = ligand.positions.copy() if len(ligand) else None
    binding_residues = _binding_site_residues(u, ligand, cfg.binding_site_cutoff_A)
    positive, negative = _charged_groups(u)

    rows = []
    exported = []
    frames_dir = Path(export_frames_dir) if export_frames_dir else None
    if frames_dir:
        frames_dir.mkdir(parents=True, exist_ok=True)

    sampled = list(range(0, len(u.trajectory), max(1, int(cfg.stride))))
    for out_i, frame_idx in enumerate(sampled):
        u.trajectory[frame_idx]
        ts = u.trajectory.ts
        time_ns = float(getattr(ts, "time", frame_idx) or frame_idx) / 1000.0
        row = {
            "frame": int(frame_idx),
            "time_ns": round(time_ns, 6),
            "protein_rmsd_A": round(_rmsd(protein.positions, ref_protein), 4),
            "radius_of_gyration_A": round(float(protein.radius_of_gyration()), 4),
            "ligand_atom_count": int(len(ligand)),
            "binding_site_residue_count": int(len(binding_residues)),
            "salt_bridge_count": _salt_bridge_count(positive, negative, cfg.salt_bridge_cutoff_A),
            "hbond_approx_count": _hbond_approx_count(u, ligand, cfg.hbond_distance_A),
        }
        if len(ligand) and ref_ligand is not None:
            row["ligand_rmsd_A"] = round(_rmsd(ligand.positions, ref_ligand), 4)
            row["ligand_protein_min_distance_A"] = round(_min_distance(protein, ligand), 4)
        else:
            row["ligand_rmsd_A"] = np.nan
            row["ligand_protein_min_distance_A"] = np.nan
        row.update(_residue_contact_summary(ligand, binding_residues, cfg.contact_cutoff_A, cfg.max_residue_contacts))
        row.update(_water_bridge_summary(u, ligand, cfg.water_selection, cfg.water_bridge_cutoff_A))
        row.update(_pi_interaction_summary(u, ligand, cfg.ligand_pi_selection, cfg.pi_centroid_cutoff_A, cfg.pi_parallel_angle_max_deg))
        rows.append(row)

        if frames_dir and (out_i in {0, len(sampled)//2, len(sampled)-1}):
            name = frames_dir / f"representative_frame_{frame_idx}.pdb"
            u.atoms.write(str(name))
            exported.append(str(name))

    meta = {
        "topology": str(topology_path),
        "trajectory": str(trajectory_path),
        "n_frames_total": int(len(u.trajectory)),
        "n_frames_sampled": int(len(rows)),
        "stride": int(cfg.stride),
        "protein_selection": cfg.protein_selection,
        "ligand_selection": cfg.ligand_selection,
        "water_selection": cfg.water_selection,
        "ligand_pi_selection": cfg.ligand_pi_selection or "auto/fallback only",
        "binding_site_cutoff_A": cfg.binding_site_cutoff_A,
        "contact_cutoff_A": cfg.contact_cutoff_A,
        "water_bridge_cutoff_A": cfg.water_bridge_cutoff_A,
        "pi_centroid_cutoff_A": cfg.pi_centroid_cutoff_A,
        "notes": [
            "hbond_approx_count is based on close N/O/S contacts unless replaced by a dedicated H-bond analysis table.",
            "water_bridge_count requires explicit water oxygen atoms simultaneously close to ligand polar atoms and protein polar atoms.",
            "pi_interaction_count uses centroid and plane-angle geometry; best results require ligand_pi_selection for the ligand aromatic ring.",
            "Residue contacts are minimum heavy-atom distances between binding-site residues and ligand selection.",
        ],
        "exported_frames": exported,
    }

    df = pd.DataFrame(rows)

    try:
        u.trajectory.close()
    except Exception:
        pass

    return df, meta


def analyze_uploaded_raw_md(topology_bytes: bytes, topology_name: str, trajectory_bytes: bytes, trajectory_name: str, config: FeatureConfig | None = None) -> tuple[pd.DataFrame, dict]:
    with tempfile.TemporaryDirectory() as td:
        top = Path(td) / _safe_name(topology_name)
        traj = Path(td) / _safe_name(trajectory_name)
        frames = Path(td) / "frames"
        top.write_bytes(topology_bytes)
        traj.write_bytes(trajectory_bytes)
        df, meta = extract_raw_md_features(top, traj, config=config, export_frames_dir=frames)
        # Explicitly release any MDAnalysis file handles (memory-mapped / file-based
        # trajectory readers) before the TemporaryDirectory context exits.  On
        # Windows, open handles prevent deletion of the temp files, causing a
        # PermissionError [WinError 32].  Closing the module-level MDA Universe
        # objects (if any) that were created inside extract_raw_md_features is not
        # straightforward from here, so we force garbage collection to trigger
        # __del__ on any lingering Universe/Reader objects and then attempt a
        # best-effort handle release via the MDAnalysis trajectory reader API.
        try:
            import gc
            gc.collect()
        except Exception:
            pass
        return df, meta
