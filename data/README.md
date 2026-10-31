# Data Records

Large arrays, WAVs, model weights and ZIPs are distributed through Zenodo, not
normal Git history. Revised identifier:
[10.5281/zenodo.23795076](https://doi.org/10.5281/zenodo.23795076).
Publication/download state was not confirmed at repository preparation.

- `release_2.0.0-rc1/`: exact small upload files, license, correction notice and
  SHA-256 manifest. Preparation-state fields are not a live record status.
- `schema/`: exact benchmark class/profile maps, parameters and environment;
  metadata, not the arrays themselves.
- `frozen_source_mapping.json`: archive paths and hashes mapped into this
  checkout; compare with `repository_source_mapping.json` for adaptations.
- `repository_source_mapping.json`: current source hashes and byte-identical
  versus adapted status relative to the frozen scientific snapshot.

The revised array is `(163200, 2048)` complex64. `X.npy` alone is 2,673,868,928
bytes including its header. The historical 139-label array is a different
`(1334400, 2048)` artifact; compressed RAR size is not an array-shape check.
Historical version: [10.5281/zenodo.20755739](https://doi.org/10.5281/zenodo.20755739).
The historical visible file inventory does not establish public historical WAV
access. The revised source ZIP separately supplies 33 screened new WAV sets.

Run `python scripts/verify_download.py --root PATH_TO_DOWNLOADS` to check the
eleven-file upload package. See the root README for extraction locations,
fixed split counts and reproduction commands. These ZIPs are not NumPy byte
shards; do not use the historical shard-restoration helper to unpack them.
