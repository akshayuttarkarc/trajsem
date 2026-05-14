# TrajSem Research: installation and feature guide

This app runs locally on Windows, macOS, and Linux. The UI opens in your browser, but the analysis stays on your machine.

## 1. Install Python

Install Python 3.10, 3.11, or 3.12 from python.org or through Conda/Mambaforge. Python 3.11 is the safest default.

Check the install:

```bash
python --version
python -m pip --version
```

On some macOS/Linux systems, use `python3` instead of `python`.

## 2. Unzip the app

Unzip the downloaded package and enter the folder:

```bash
cd trajsem_research
```

## 3. Install the lightweight UI

This installs the local desktop-style app, demo mode, CSV mode, plots, reports, verifier, and expert annotation workflow.

```bash
python -m pip install -r requirements.txt
```

## 4. Run the app

```bash
python run_app.py
```

A Streamlit URL appears in the terminal, usually `http://localhost:8501`. Open it in the browser if it does not open automatically.

## 5. Run without input files

Choose **No-input demo** in the sidebar. This gives a complete demonstration: descriptors, semantic events, grounded report, verifier, LLM chat tab, and exports.

## 6. Enable raw MD trajectory analysis

For real GROMACS/AMBER/NAMD-style trajectory files, install the MD stack:

```bash
python -m pip install -r requirements-md.txt
python run_app.py
```

Then choose **Raw MD trajectory** and upload both:

- topology/structure: `.pdb`, `.gro`, `.psf`, `.prmtop`, `.mol2`, `.pqr`, or `.tpr`
- trajectory: `.xtc`, `.trr`, `.dcd`, `.nc`, `.netcdf`, or `.pdb`

## 7. Optional pocket tools

The app can ingest pocket-volume CSV tables from fpocket, MDpocket, POVME, or custom tools. If `fpocket` is installed and on PATH, the app can also run a single-frame fpocket check from the UI.

External tools are not bundled because they are OS-specific and often installed through Conda, package managers, or source builds.

## 8. Optional local LLM chat/reporting

Install and start Ollama locally, then pull a model such as `llama3.1`:

```bash
ollama pull llama3.1
```

In the app, enable **Use local Ollama if available**. The report and chat will use the local model. If Ollama is unavailable, TrajSem uses deterministic evidence-grounded text generation.

## Extracted features

Core structural descriptors:

- `protein_rmsd_A`
- `ligand_rmsd_A`
- `radius_of_gyration_A`
- `ligand_protein_min_distance_A`
- `binding_site_residue_count`

Residue-level interaction geometry:

- `contact_<RESIDUE>_distance_A`
- `contact_<RESIDUE>_present`
- residue contact loss events
- residue-ligand separation events

Hydrogen-bond / polar-contact screening:

- `hbond_approx_count`
- persistent hydrogen-bond occupancy columns when supplied as CSV, for example `hbond_ASP147_occupancy`

Salt bridges:

- `salt_bridge_count`

Water bridges:

- `water_bridge_count`
- `water_bridge_residue_count`
- `water_bridge_residues`

Definition: a bridging water is counted when a water oxygen is within the configured cutoff of at least one ligand polar atom and at least one protein polar atom in the same frame.

π interactions:

- `pi_interaction_count`
- `pi_interaction_residues`
- `pi_validation_mode`
- `pi_min_centroid_distance_A`
- `pi_min_plane_angle_deg`

Best practice: provide an explicit ligand aromatic ring selection, for example:

```text
resname LIG and name C1 C2 C3 C4 C5 C6
```

Without an explicit ligand ring selection, the app only uses a conservative planar-ligand fallback when possible.

Pocket and cryptic-pocket descriptors:

- `pocket_volume_A3`
- `binding_site_sasa_A2`
- `loop_*_displacement_A`
- `pocket_lining_residue_count` if supplied by external pocket analysis

Cryptic-pocket claims are now stricter. A pocket-volume spike alone is reported only as a volume-only candidate. A stronger cryptic-pocket candidate requires pocket-volume expansion plus independent support such as SASA increase, loop displacement, or lining-residue evidence.

## Events the app can report

- ligand pose drift
- persistent hydrogen-bond signal
- hydrogen-bond network increase
- residue contact loss
- residue-ligand separation
- water-bridge network
- π-interaction candidate
- salt/polar interaction changes when represented in descriptors
- transient pocket-opening candidate
- conformational shift / loop displacement
- binding-site exposure increase

## LLM chat questions you can ask

Examples:

```text
What features were extracted from this trajectory?
Which events are strongest and why?
Is the ligand stable?
Which residues lose contact with the ligand?
Are there water bridges and when do they occur?
Are there π interactions and are they well validated?
Is the cryptic-pocket claim publication-ready?
Which claims are weak or unsupported?
What should I include in the Results section?
What follow-up analyses are needed before submission?
```

The chat is evidence-constrained. If the required descriptor was not extracted or uploaded, it should say what is missing rather than inventing an answer.

## Exported research artifacts

The Exports tab provides:

- `trajsem_descriptors.csv`
- `trajsem_events.json`
- `trajsem_report.md`
- `trajsem_verification.json`
- `trajsem_povme_template.cfg`
- expert annotation template

## Publication caveats

This is a serious research scaffold, not a validated black-box scientific instrument. Before paper submission, run replicate trajectories, inspect representative frames, calibrate thresholds for your protein family, and compare against blinded expert annotations.
