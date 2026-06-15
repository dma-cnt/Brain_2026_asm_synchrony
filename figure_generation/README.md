# Figure generation

This folder contains scripts and intermediate assets used to build the revised manuscript figures.

Run scripts from the repository root with the `cnt` Conda environment:

```bash
conda run -n cnt python figure_generation/figure_1.py
conda run -n cnt python figure_generation/clinical_exposure_state_analysis_brain_figure.py
conda run -n cnt python figure_generation/figure_3.py
conda run -n cnt python figure_generation/figure_4.py
conda run -n cnt python figure_generation/supplementary_figure_1.py
```

Default generated outputs go to `figure_generation/outputs/` so the submission folder remains clean. Use `--output-dir revised_figures` only when intentionally regenerating the final TIFF files in place.
