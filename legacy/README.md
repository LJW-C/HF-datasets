# Historical Generators

`build_datasets.py` and `build_all_dataset.py` here preserve the previously
root-level generators. They are **not** the corrected 2.0.0-rc1 pipeline.
They do not establish source identity, a raw-time-disjoint split or the
corrected output-PSD Doppler width.

Use root `build_datasets.py` / `build_all_dataset.py` entry points or
`revision_v2/build_verified_dataset.py` for new verified-source data.
Historical recordings and arrays are not silently overwritten or corrected.
Old all-modem acquisition remains available for inspection, but is not proof
of 139 independently verified modes.
