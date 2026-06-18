# IEEE Dataset Paper Draft

This folder contains IEEE-style LaTeX drafts for the HF Watterson radio signal dataset generator.

Files:

```text
ieee_dataset_paper.tex
hf_digital_mode_dataset_ieee_cited.tex
references.bib
figures/
tables/
```

Compile with a local LaTeX installation:

```bash
pdflatex hf_digital_mode_dataset_ieee_cited.tex
bibtex hf_digital_mode_dataset_ieee_cited
pdflatex hf_digital_mode_dataset_ieee_cited.tex
pdflatex hf_digital_mode_dataset_ieee_cited.tex
```

Before compiling the paper with the latest benchmark results, run:

```bash
python ../make_paper_results.py --runs-dir ../runs/model_suite --paper-dir .
```

Before submission, update:

```text
author names
affiliations
repository URL
dataset DOI or download URL
acknowledgments
```

The cited paper draft imports measured benchmark results from `tables/` and `figures/`. Re-run `make_paper_results.py` whenever the model results change.
