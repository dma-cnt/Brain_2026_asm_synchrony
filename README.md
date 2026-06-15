# ASM synchrony paper analysis code

This repository contains analysis and figure-generation code for the Brain manuscript:

> Intracranial EEG synchrony tracks antiseizure medication load in drug-resistant epilepsy

## Repository layout

- `brain_r2/`: current revision analyses, statistical checks, and source outputs used by the revised manuscript.
- `brain_r1/`: earlier figure notebooks and static panel assets still needed to reproduce Figure 1.
- `figure_generation/`: scripts and intermediate assets used to generate the final revised figures.
- `figure_generation/outputs/`: preview PNGs and companion summary files produced while building figures.
- `revised_figures/`: final submission-ready TIFF files only.

Legacy exploratory notebooks, iEEG login binaries, and editorial/reviewer correspondence extracts have been removed from the public tree. Local copies are kept under ignored `local_private/` if this cleanup was run in the original workspace.

## Environment

Create the Conda environment from `environment.yml`:

```bash
conda env create -f environment.yml
conda activate cnt
```

All project scripts should be run in the `cnt` environment.

## Data

The scripts expect the private analysis data directory to be available separately. Set `CNT_DATA_PATH` to the local `cnt-data` directory before running scripts:

```bash
export CNT_DATA_PATH=/path/to/cnt-data
```

Raw intracranial EEG and medication data are not stored in this repository because they contain protected clinical information and access-controlled iEEG.org material.

## Reproducing figures

Run the figure scripts from the repository root:

```bash
conda run -n cnt python figure_generation/figure_1.py
conda run -n cnt python figure_generation/clinical_exposure_state_analysis_brain_figure.py
conda run -n cnt python figure_generation/figure_3.py
conda run -n cnt python figure_generation/figure_4.py
conda run -n cnt python figure_generation/supplementary_figure_1.py
```

By default, generated working outputs are written to `figure_generation/outputs/`. The submitted TIFFs are checked in under `revised_figures/`.

To regenerate the final submission TIFFs in place, pass `--output-dir revised_figures` to each figure script. This will also write preview/supporting files for scripts that produce them, so move those back to `figure_generation/outputs/` before publication if you want `revised_figures/` to remain submission-only.
