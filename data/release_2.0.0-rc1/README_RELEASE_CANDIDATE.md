# HFDigiSet 2.0.0-rc1

Version identifier: https://doi.org/10.5281/zenodo.23795076

Creator confirmed for this dataset record: Junwei Li. The author list of the
associated manuscript is separate and is not changed by this dataset metadata.
Data license: CC BY 4.0. Software license: MIT where indicated; third-party
software retains its own terms.

## Identifier and publication state

The version DOI was reserved by the record owner. At metadata preparation,
public release had not been confirmed. A reserved DOI alone is not evidence
that files are publicly downloadable. Consult the version's Zenodo landing
page for its current publication/access state. Preparation timestamps and code
content hashes are not a publication date or public Git commit.

The historical v1 record is https://doi.org/10.5281/zenodo.20755739. Its files
are dataset.rar (historical 17-class arrays), datasets.rar (139 historical
candidate labels), and iq_catalog.pdf. Those arrays are not this corrected
benchmark. The visible v1 file list and inspected local namesake RAR inventories
do not establish public access to the historical source WAVs.

## Scope

- 17 revised configurations, nine parent families; the 17 labels are fine
  configurations, not 17 parent families or a grouping of the historical array.
- 163200 vectors, each 2048 complex64 samples at 6000 Hz.
- Eight target SNRs: -10, -5, 0, 5, 10, 15, 20, 25 dB, over the full complex
  sampling bandwidth, not equal in-band SNR across configurations.
- 1200 vectors per configuration/SNR TOTAL across four profiles. Exact fixed
  split: 114240 train, 24480 validation, 24480 test.
- Profile IDs: 0 good, 1 moderate, 2 poor, 3 flutter. Each configuration/SNR/
  profile contributes 210/45/45 train/validation/test vectors.
- 33 screened source WAV/payload/control records: the 17 benchmark sources and
  16 additional recordings without regenerated-array validation.
- Three OFDM selections are excluded after observed BPSK31 fallback; 103 other
  historical source entries still require complete review. This version is not
  a corrected, fully validated 139-class release.

Sources were recorded with Fldigi 4.2.11: one 60 s, 16 kHz, mono PCM-24 source
per configuration, with modem/carrier readbacks and transmitter monitoring.
Readbacks and spectrum screening are not independent RSID/payload decoding.

Raw intervals [3,42.2), [42.2,50.6), [50.6,59) seconds are processed separately
before Hilbert conversion, downmixing, resampling and fading. Each main-benchmark
partition has a 0.5 s guard at both ends. One recording and one payload per
class remain shared across partitions: this is raw-time-disjoint, not
independent-recording, independent-message or OTA validation.

## Upload files and their contents

1. benchmark_17_2.0.0-rc1.zip: revised arrays, label maps, exact split indices,
   raw partitions, per-vector provenance, generation environment and snapshots.
2. screened_sources_33_2.0.0-rc1.zip: raw_verified/ contains the 17 benchmark
   recordings; raw_priority/ contains 16 additional screened recordings and
   records of unsuccessful requests. Failure records are not successful WAVs.
3. code_snapshot_2.0.0-rc1.zip: exact scientific code, observed dependencies,
   documentation and the compact Figure 2 plotting code/asset. The base Git
   commit by itself does not identify this working-tree snapshot.
4. diagnostics_and_results_2.0.0-rc1.zip: reference predictions/configs/indices/
   histories/QC; repeated-training records; and the separate paired-duration
   diagnostic's arrays, labels and results.
5. source_wav_metadata.csv: source IDs, class IDs where applicable, source
   paths, formats, controls, payload IDs and SHA-256 values for the 33 new WAVs.
6. data_dictionary.json: field semantics and units. clean_power.npy is faded
   crop power BEFORE normalization, not stored received-vector power.
7. dataset_manifest.json: version, reserved DOI, creator, code content hash,
   publication state at preparation and scope.
8. SHA256SUMS.json: sizes and SHA-256 values for the public upload files other
   than this checksum file itself.
9. LICENSE_DATA.txt: data-license and attribution statement.
10. DATA_CORRECTION_NOTICE.md: historical identity/channel issues and limits.
11. README_RELEASE_CANDIDATE.md: this version-specific description and commands.

## Extract and reproduce

Start with an empty working folder. Extract the code ZIP there, retaining its
sibling revision_v2/ and hf-digital-mode-iq-dataset-github/ directories.
Extract the benchmark ZIP into revision_v2/benchmark_17/.
Extract the source ZIP into revision_v2/.
Extract diagnostics_and_results ZIP into revision_v2/ if inspecting/recomputing
the frozen predictions. It also contains paired_windows_20261030/ and archived
run directories. Do not train over these archived directories.

Use the observed requirements-recorded.txt and generation/training environment
records. The recorded neural environment was Python 3.9.19, PyTorch
2.0.1+cu117 and an RTX 4090; this is not a promise of cross-platform bitwise
identity. Historical acquisition paths in JSON are retained; the generator
resolves the adjacent source.wav first.

```powershell
python revision_v2/verify_dataset.py revision_v2/benchmark_17
python revision_v2/train_suite.py --dataset-root revision_v2/benchmark_17 --out-dir revision_v2/reproduced_runs --models conformer --seeds 20261029,20261030,20261031 --jobs 1
python revision_v2/train_classical_baseline.py --dataset-root revision_v2/benchmark_17 --out-dir revision_v2/reproduced_rf/seed_20261029 --seed 20261029
python revision_v2/build_verified_dataset.py --raw-root revision_v2/raw_verified --out-root revision_v2/rebuilt_benchmark --seed 20261029
```

The seven-model reference comparison uses training seed 20261029. Only
Conformer and Random Forest have three-seed 17-class results: test accuracy
85.37 +/- 0.10 and 63.76 +/- 0.07 percent, respectively (sample SD across seeds).
The reference-seed Conformer accuracy remains 85.26%, not the multi-seed mean.
No architecture-wide significance or independent-session confidence is claimed.

The paired duration diagnostic uses four Olivia/Contestia configurations,
12800 vectors per arm (8960/1920/1920 split), and 400 vectors per configuration/
SNR. IQ-CNN and Random Forest are trained for three seeds at 2048 and 8192
points. Both arms share crop starts, channel realizations, CFO, phase and the
underlying noise, with separate clean-power normalization and a common
8192-point guard. This is not a Conformer duration experiment, a new 17-class
test score or a proof of coding-related causality.

To plot the article's compact four-mode quality-control figure, after extracting
the arrays and diagnostics as above:

```powershell
python validation_figures/plot_compact_examples.py
```

This creates validation_figures/figures/figure5.pdf, the historical asset name
used for Figure 2 in the article. Sample indices are in
revision_v2/results/verified_examples.json. No manuscript or reviewer letter
is included in this data upload.

To validate the upload files themselves (not the extracted array tree):

```powershell
python revision_v2/verify_release_candidate.py --root PATH_TO_THE_UPLOAD_FILES
```

The pre-reservation prepare_release_candidate.py script is a historical packer,
not a command to overwrite this DOI-bound package. Use fresh output folders
for all scientific reruns.
