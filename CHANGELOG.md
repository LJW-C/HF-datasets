# Changes

## Portable scientific-code update - 2026-10-31

- Align the repository title, version scope and citation with the revised
  `2.0.0-rc1` benchmark and reserved DOI `10.5281/zenodo.23795076`.
- Separate 17 validated configurations, 33 screened sources and the historical
  139-label catalog. Retain exclusions, unaudited-source status and limitations.
- Add frozen corrected channel/generation sources and predefined-split support.
  Archive old root builders in `legacy/`; root aliases use the corrected builder.
- Add explicit source-device selection, portable training entry points, safe
  checkpoint loading, no-overwrite checks and headless visualization.
- Include measured reference/repeat/duration CSVs and reference plots, with
  original seed/test-split distinctions. No full experiment was rerun here.
- Add checksum verification, synthetic regression and CPU smoke tests, and
  repair CI references to removed historical scripts.
- Record original versus adapted code hashes. Keep Zenodo-bound archive bytes
  and frozen metadata unchanged; this commit is not inserted into old manifests.

The revised record's public availability was not independently confirmed during
this update. No newly generated 139-class arrays or independently decoded
RSID/payload identity validation is claimed.
