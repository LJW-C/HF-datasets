# Data correction notice for the historical HFDigiSet release

The historical 139-configuration arrays and recordings have been preserved for
traceability. They must not be interpreted as 139 independently authenticated
waveform configurations.

## OFDM labels

On the locally installed Fldigi 4.2.11, requesting OFDM3500, OFDM500F or OFDM750F
through either XML-RPC name or numeric ID selects BPSK31 instead. The requested
names still appear in modem.get_names(). The recorded historical OFDM WAVs have
approximately 46 Hz occupied bandwidth, consistent with this fallback, rather
than the requested OFDM configuration.

Official Fldigi 4.2.13 source was downloaded from:
https://www.w1hkj.org/files/fldigi/fldigi-4.2.13.tar.gz

Its `src/dialogs/fl_digi.cxx` contains an OFDM initialization block commented out
with the explanation "Commented out pending development work by KL4YFD". This
source evidence is consistent with the local 4.2.11 runtime observations; it is
not a claim that the two versions have identical source code everywhere.

The three historical OFDM labels are excluded from the verified v2 sources.
They have NOT been silently relabeled, accepted on the basis of the requested
name, or replaced with an unrelated Python-generated waveform. See
`ofdm_identity_probe.json` for exact observed results. A future OFDM source
requires a working verified transmitter implementation and new recordings.

## Other sources

The 17 benchmark modes were recorded anew with actual-modem readback, disabled
and read-back TX/RX RSID, saved payloads and spectrum checks. A further 16 flagged
non-OFDM modes were recorded anew. Their verified WAVs are in `raw_verified/` and
`raw_priority/`. The raw_priority first-pass manifest preserves three OFDM
failures; those entries do not represent successful recordings.

The revised 17-class arrays use these new benchmark WAVs and a raw-time-disjoint
split with independently simulated fading per split. The historical 139-class
arrays were not overwritten or claimed to have been fully regenerated. The
other historical source configurations still require complete source-quality
review before a corrected full-catalog release can be declared validated.

## Source format change

New recordings: 60 seconds, 16 kHz, mono PCM-24. Historical recordings: 180
seconds, 8 kHz, mono PCM-16. The paper and new release metadata must describe
these separately. A 17-class subset is a separate generated dataset, not an
assertion of row identity with the historical 139-class arrays.

## Gaussian Doppler shaping correction

The historical generator used exp(-2*pi*pi*sigma_f*sigma_f*t*t) as the Gaussian
FIR while describing the requested Doppler spread as two standard deviations
of the output power spectrum. Filtering white noise with that amplitude
response produces a power-spectrum width approximately 1/sqrt(2) of the stated
value. The revised implementation uses coefficient 4, unit-energy FIR taps,
60 dB truncation and protected polyphase interpolation. Long-process PSD checks
are supplied in channel_validation.json. This is a numerical implementation
check, not formal ITU simulator certification.

The revised 17-class arrays and results were regenerated. Historical arrays
retain their historical channel realization; they have not been corrected by
editing a formula, label, manifest or result in place. The data version and exact
code snapshot must be identified when comparing the two experimental scopes.

The local v2 experiments are not an OTA validation. They use time-disjoint
partitions of one recording per class. RSID controls were disabled and monitored;
an independent waveform-level RSID decoder was not run.

## Historical source-WAV availability

The visible v1 file list contains dataset.rar, datasets.rar and iq_catalog.pdf.
The inspected local namesake RAR inventories do not contain source WAV files.
This does not establish a public download location for all historical WAVs.
This version separately supplies 33 new screened WAV/payload/control records;
only 17 have regenerated-array and benchmark validation.
