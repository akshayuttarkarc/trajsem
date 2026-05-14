from __future__ import annotations

import numpy as np
import pandas as pd


def make_demo(seed: int = 7, n_frames: int = 1001) -> pd.DataFrame:
    """Create a realistic-looking descriptor table for a protein-ligand MD run."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 100, n_frames)

    protein_rmsd = 1.25 + 0.25 * np.tanh((t - 18) / 10) + 0.08 * np.sin(t / 7) + rng.normal(0, 0.035, n_frames)
    ligand_rmsd = 1.1 + 0.18 * np.sin(t / 8) + rng.normal(0, 0.07, n_frames)
    ligand_rmsd += np.where(t > 55, 2.2 / (1 + np.exp(-(t - 61) / 2.5)), 0)
    ligand_rmsd -= np.where(t > 78, 0.7 / (1 + np.exp(-(t - 82) / 2.5)), 0)

    pocket_volume = 135 + 18 * np.sin(t / 9) + rng.normal(0, 8, n_frames)
    pocket_volume += 245 * np.exp(-0.5 * ((t - 69) / 6.2) ** 2)

    hbond_asp147 = np.clip(0.22 + 0.53 / (1 + np.exp(-(t - 39) / 3.8)) + rng.normal(0, 0.06, n_frames), 0, 1)
    hbond_lys203 = np.clip(0.04 + 0.62 / (1 + np.exp(-(t - 70) / 3.0)) + rng.normal(0, 0.07, n_frames), 0, 1)
    contact_phe91 = np.clip(0.78 - 0.58 / (1 + np.exp(-(t - 57) / 2.5)) + rng.normal(0, 0.06, n_frames), 0, 1)
    loop_185_193 = 2.1 + 1.7 * np.exp(-0.5 * ((t - 68) / 7.5) ** 2) + rng.normal(0, 0.09, n_frames)
    sasa_binding_site = 510 + 95 * np.exp(-0.5 * ((t - 69) / 8.5) ** 2) + rng.normal(0, 13, n_frames)
    rg = 21.4 + 0.05 * np.sin(t / 10) + rng.normal(0, 0.025, n_frames)
    water_bridge = np.clip(0.10 + 0.75 * np.exp(-0.5 * ((t - 44) / 9.0) ** 2) + rng.normal(0, 0.05, n_frames), 0, 1)
    water_bridge_count = (water_bridge * 3.2).round().astype(int)
    pi_count = np.where((t > 18) & (t < 52), 1, 0)
    pi_count = np.where((t > 70) & (t < 86), 1, pi_count)
    salt_bridge = np.clip(3 + rng.integers(-1, 2, n_frames), 0, None).astype(int)
    pi_min_d = np.where(pi_count > 0, 4.6 + rng.normal(0, 0.12, n_frames), 7.2 + rng.normal(0, 0.25, n_frames))
    pi_angle = np.where(pi_count > 0, 18 + rng.normal(0, 4, n_frames), 54 + rng.normal(0, 8, n_frames))

    return pd.DataFrame({
        "time_ns": t.round(3),
        "protein_rmsd_A": protein_rmsd.round(3),
        "ligand_rmsd_A": ligand_rmsd.round(3),
        "pocket_volume_A3": pocket_volume.round(2),
        "hbond_asp147_occupancy": hbond_asp147.round(3),
        "hbond_lys203_occupancy": hbond_lys203.round(3),
        "contact_phe91_occupancy": contact_phe91.round(3),
        "loop_185_193_displacement_A": loop_185_193.round(3),
        "binding_site_sasa_A2": sasa_binding_site.round(2),
        "radius_of_gyration_A": rg.round(3),
        "salt_bridge_count": salt_bridge,
        "water_bridge_count": water_bridge_count,
        "water_bridge_residue_count": np.where(water_bridge_count > 0, 1, 0),
        "water_bridge_residues": np.where(water_bridge_count > 0, "ASP147;LYS203", ""),
        "pi_interaction_count": pi_count,
        "pi_interaction_residues": np.where(pi_count > 0, "PHE91", ""),
        "pi_validation_mode": np.where(pi_count > 0, "user_selection", "user_selection"),
        "pi_min_centroid_distance_A": pi_min_d.round(3),
        "pi_min_plane_angle_deg": pi_angle.round(2),
    })
