"""
Automatically generate raw WAV recordings from Fldigi transmit modes.

This script controls Fldigi through XML-RPC, sends random text payloads, records
audio from a selected sound device, and saves one WAV file per radio mode.

Typical use:
    python generate_fldigi_wavs.py --device-id 7 --out-root "E:\\仿真\\raw_wav"

Before running:
    1. Start Fldigi.
    2. Enable XML-RPC in Fldigi, usually at http://127.0.0.1:7362.
    3. Route Fldigi Playback to a virtual cable input.
    4. Set --device-id to the corresponding virtual cable output recording device.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import string
import time
import xmlrpc.client
from datetime import datetime
from pathlib import Path

import numpy as np


DEFAULT_FLDIGI_XMLRPC_URL = "http://127.0.0.1:7362"

DEFAULT_MODES = [
    {"folder": "BPSK31", "fldigi_modem": "BPSK31"},
    {"folder": "BPSK63", "fldigi_modem": "BPSK63"},
    {"folder": "BPSK125", "fldigi_modem": "BPSK125"},
    {"folder": "QPSK31", "fldigi_modem": "QPSK31"},
    {"folder": "QPSK63", "fldigi_modem": "QPSK63"},
    {"folder": "QPSK125", "fldigi_modem": "QPSK125"},
    {"folder": "RTTY", "fldigi_modem": "RTTY"},
    {"folder": "MFSK16", "fldigi_modem": "MFSK16"},
    {"folder": "MFSK32", "fldigi_modem": "MFSK32"},
    {"folder": "Olivia_4_250", "fldigi_modem": "OLIVIA-4/250"},
    {"folder": "Olivia_8_500", "fldigi_modem": "OLIVIA-8/500"},
    {"folder": "Contestia_4_250", "fldigi_modem": "Cont-4/250"},
    {"folder": "Contestia_8_500", "fldigi_modem": "Cont-8/500"},
    {"folder": "DominoEX_8", "fldigi_modem": "DOMEX8"},
    {"folder": "Thor_16", "fldigi_modem": "THOR16"},
    {"folder": "Throb_2", "fldigi_modem": "THROB2"},
    {"folder": "CW", "fldigi_modem": "CW"},
]

EXPECTED_OBW_HZ = {
    "Olivia_4_250": 250.0,
    "Olivia_8_500": 500.0,
    "Contestia_4_250": 250.0,
    "Contestia_8_500": 500.0,
}


def import_audio_packages():
    try:
        import sounddevice as sd
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError(
            "Missing audio dependency. Install requirements first:\n"
            "    pip install -r requirements.txt"
        ) from exc
    return sd, sf


def import_signal_package():
    try:
        from scipy import signal
    except ImportError as exc:
        raise RuntimeError(
            "Missing scipy dependency for recording validation.\n"
            "Install requirements first:\n"
            "    pip install -r requirements.txt"
        ) from exc
    return signal


def make_random_payload(num_chars: int) -> str:
    alphabet = string.ascii_uppercase + string.ascii_lowercase + string.digits + "     .,;:!?-\n"
    random_text = "".join(random.choice(alphabet) for _ in range(num_chars))
    header = (
        "THIS IS A DATASET GENERATION PAYLOAD FOR DIGITAL RADIO MODE CLASSIFICATION.\n"
        "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG 0123456789.\n"
    )
    return header + random_text


def connect_fldigi(url: str):
    fldigi = xmlrpc.client.ServerProxy(url, allow_none=True)
    try:
        name = fldigi.fldigi.name()
        version = fldigi.fldigi.version()
        print(f"Connected to Fldigi: {name}, version: {version}")
    except Exception as exc:
        raise RuntimeError(
            "Cannot connect to Fldigi XML-RPC.\n"
            "Please start Fldigi and confirm the XML-RPC port is enabled.\n"
            f"XML-RPC URL: {url}\n"
            f"Original error: {exc}"
        ) from exc
    return fldigi


def print_available_modems(fldigi) -> None:
    try:
        names = fldigi.modem.get_names()
        print("\nFldigi modem names:")
        for name in names:
            print(f"  {name}")
        print()
    except Exception as exc:
        print(f"Could not read Fldigi modem list: {exc}")


def list_audio_devices() -> None:
    sd, _ = import_audio_packages()
    print(sd.query_devices())


def estimate_baseband_obw(
    audio: np.ndarray,
    fs: int,
    carrier_hz: float,
    max_seconds: float = 18.0,
) -> dict[str, float]:
    signal = import_signal_package()
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    x = x[: min(len(x), int(round(max_seconds * fs)))]
    x = x - float(np.mean(x))

    analytic = signal.hilbert(x)
    n = np.arange(len(analytic), dtype=np.float64)
    baseband = analytic * np.exp(-1j * 2.0 * np.pi * carrier_hz * n / fs)

    nperseg = min(4096, max(256, len(baseband)))
    freqs, psd = signal.welch(
        baseband,
        fs=fs,
        nperseg=nperseg,
        noverlap=nperseg // 2,
        return_onesided=False,
        scaling="density",
    )
    freqs = np.fft.fftshift(freqs)
    psd = np.fft.fftshift(psd)
    mask = np.abs(freqs) <= 1400.0
    f = freqs[mask]
    p = psd[mask]
    order = np.argsort(f)
    f = f[order]
    p = p[order]
    cumulative = np.cumsum(p)
    cumulative = cumulative / cumulative[-1]
    f005 = float(np.interp(0.005, cumulative, f))
    f995 = float(np.interp(0.995, cumulative, f))
    f025 = float(np.interp(0.025, cumulative, f))
    f975 = float(np.interp(0.975, cumulative, f))
    return {
        "peak_hz": float(f[np.argmax(p)]),
        "obw_95_hz": f975 - f025,
        "obw_99_hz": f995 - f005,
    }


def validate_obw(folder_name: str, obw_99_hz: float) -> str:
    expected = EXPECTED_OBW_HZ.get(folder_name)
    if expected is None:
        return "unchecked"
    return "pass" if 0.70 * expected <= obw_99_hz <= 1.35 * expected else "fail"


def backup_existing_wav(path: Path) -> str | None:
    if not path.exists():
        return None
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = path.with_name(f"{path.stem}.bak_{stamp}{path.suffix}")
    shutil.copy2(path, backup_path)
    print(f"Existing WAV backed up to: {backup_path}")
    return str(backup_path)


def configure_special_mode_params(fldigi, folder_name: str, modem_name: str) -> None:
    # Exact modem names such as OLIVIA-4/250 and Cont-4/250 already encode the
    # tones/bandwidth. These calls are only a fallback for older Fldigi versions.
    if modem_name not in {"OLIVIA", "CONTESTIA"}:
        return
    try:
        if folder_name == "Olivia_4_250":
            fldigi.modem.olivia.set_tones(4)
            fldigi.modem.olivia.set_bandwidth(250)
        elif folder_name == "Olivia_8_500":
            fldigi.modem.olivia.set_tones(8)
            fldigi.modem.olivia.set_bandwidth(500)
        elif folder_name == "Contestia_4_250":
            fldigi.modem.olivia.set_tones(4)
            fldigi.modem.olivia.set_bandwidth(250)
        elif folder_name == "Contestia_8_500":
            fldigi.modem.olivia.set_tones(8)
            fldigi.modem.olivia.set_bandwidth(500)
    except Exception as exc:
        print(f"Warning: automatic tones/bandwidth setup failed for {folder_name}: {exc}")
        print("Please confirm these settings manually in Fldigi if this mode is used.")


def prepare_fldigi_tx(fldigi, modem_name: str, folder_name: str, carrier_hz: float, payload: str) -> None:
    print(f"Setting Fldigi modem: {modem_name}")

    try:
        fldigi.main.rx()
    except Exception:
        pass

    old_modem = fldigi.modem.set_by_name(modem_name)
    time.sleep(0.5)

    configure_special_mode_params(fldigi, folder_name, modem_name)

    try:
        fldigi.modem.set_carrier(int(round(carrier_hz)))
    except Exception as exc:
        print(f"Warning: failed to set carrier. You can manually click {carrier_hz} Hz in Fldigi. Error: {exc}")

    try:
        fldigi.main.set_txid(False)
        fldigi.main.set_rsid(False)
    except Exception:
        pass

    fldigi.text.clear_tx()
    time.sleep(0.2)
    fldigi.text.add_tx(payload)
    print(f"Current modem changed from {old_modem} to {modem_name}")


def select_modes(mode_arg: str) -> list[dict[str, str]]:
    if mode_arg.lower() == "all":
        return DEFAULT_MODES
    wanted = {item.strip() for item in mode_arg.split(",") if item.strip()}
    modes = [mode for mode in DEFAULT_MODES if mode["folder"] in wanted or mode["fldigi_modem"] in wanted]
    missing = sorted(wanted - {mode["folder"] for mode in modes} - {mode["fldigi_modem"] for mode in modes})
    if missing:
        raise ValueError(f"Unknown mode(s): {', '.join(missing)}")
    return modes


def record_one_mode(
    fldigi,
    mode: dict[str, str],
    out_root: Path,
    device_id: int,
    record_fs: int,
    duration_sec: int,
    carrier_hz: float,
    payload_chars: int,
    backup_existing: bool,
) -> dict[str, object]:
    sd, sf = import_audio_packages()

    folder_name = mode["folder"]
    modem_name = mode["fldigi_modem"]
    out_dir = out_root / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{folder_name}.wav"
    backup_path = backup_existing_wav(out_path) if backup_existing else None

    payload = make_random_payload(payload_chars)
    prepare_fldigi_tx(fldigi, modem_name, folder_name, carrier_hz, payload)

    print("\n" + "=" * 72)
    print(f"Recording mode: {folder_name}")
    print(f"Output WAV:     {out_path}")
    print("=" * 72)

    audio = sd.rec(
        int(duration_sec * record_fs),
        samplerate=record_fs,
        channels=1,
        dtype="float32",
        device=device_id,
    )

    time.sleep(0.5)
    fldigi.main.tx()

    for remaining in range(duration_sec, 0, -10):
        print(f"{folder_name}: about {remaining} seconds remaining...")
        time.sleep(min(10, remaining))

    sd.wait()

    try:
        fldigi.main.rx()
    except Exception:
        pass

    sf.write(out_path, audio, record_fs)

    rms = float(np.sqrt(np.mean(audio**2)))
    peak = float(np.max(np.abs(audio)))
    print(f"Saved: {out_path}")
    print(f"RMS={rms:.6f}, Peak={peak:.6f}")

    if rms < 1e-4:
        print("Warning: this recording may be silent. Check Fldigi playback and recording device routing.")

    obw_stats: dict[str, float] = {}
    obw_status = "not_run"
    try:
        obw_stats = estimate_baseband_obw(audio, record_fs, carrier_hz)
        obw_status = validate_obw(folder_name, float(obw_stats["obw_99_hz"]))
        print(
            "Baseband OBW check: "
            f"OBW99={obw_stats['obw_99_hz']:.1f} Hz, "
            f"OBW95={obw_stats['obw_95_hz']:.1f} Hz, "
            f"peak={obw_stats['peak_hz']:.1f} Hz, "
            f"status={obw_status}"
        )
        if obw_status == "fail":
            expected = EXPECTED_OBW_HZ[folder_name]
            print(f"Warning: {folder_name} expected about {expected:.0f} Hz.")
    except Exception as exc:
        print(f"Warning: recording validation failed for {folder_name}: {exc}")

    time.sleep(2)
    return {
        "folder": folder_name,
        "fldigi_modem": modem_name,
        "wav_path": str(out_path),
        "record_fs": record_fs,
        "duration_sec": duration_sec,
        "rms": rms,
        "peak": peak,
        "backup_path": backup_path,
        "obw_status": obw_status,
        **obw_stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate raw WAV recordings from Fldigi digital modes.")
    parser.add_argument("--xmlrpc-url", default=DEFAULT_FLDIGI_XMLRPC_URL, help="Fldigi XML-RPC URL.")
    parser.add_argument("--device-id", type=int, default=7, help="Recording device id, e.g. VB-CABLE Output.")
    parser.add_argument("--record-fs", type=int, default=8000, help="Recording sample rate.")
    parser.add_argument("--duration-sec", type=int, default=180, help="Recording duration per mode.")
    parser.add_argument("--carrier-hz", type=float, default=1500.0, help="Fldigi audio carrier frequency.")
    parser.add_argument("--payload-chars", type=int, default=30000, help="Random text payload length.")
    parser.add_argument("--out-root", default=r"E:\仿真\raw_wav", help="Output root folder for raw WAV files.")
    parser.add_argument("--modes", default="all", help="all, or comma-separated folder/modem names.")
    parser.add_argument("--list-devices", action="store_true", help="List audio devices and exit.")
    parser.add_argument("--print-modems", action="store_true", help="Print Fldigi modem names before recording.")
    parser.add_argument("--no-backup", action="store_true", help="Do not back up an existing WAV before overwriting.")
    args = parser.parse_args()

    if args.list_devices:
        list_audio_devices()
        return

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    selected_modes = select_modes(args.modes)
    fldigi = connect_fldigi(args.xmlrpc_url)

    if args.print_modems:
        print_available_modems(fldigi)

    metadata = {
        "xmlrpc_url": args.xmlrpc_url,
        "device_id": args.device_id,
        "record_fs": args.record_fs,
        "duration_sec": args.duration_sec,
        "carrier_hz": args.carrier_hz,
        "payload_chars": args.payload_chars,
        "modes": [],
    }

    for mode in selected_modes:
        try:
            result = record_one_mode(
                fldigi=fldigi,
                mode=mode,
                out_root=out_root,
                device_id=args.device_id,
                record_fs=args.record_fs,
                duration_sec=args.duration_sec,
                carrier_hz=args.carrier_hz,
                payload_chars=args.payload_chars,
                backup_existing=not args.no_backup,
            )
            metadata["modes"].append(result)
        except Exception as exc:
            print("\n" + "!" * 72)
            print(f"Failed to record {mode['folder']}: {exc}")
            print("Check the Fldigi modem name, XML-RPC connection, and audio routing.")
            print("!" * 72 + "\n")
            try:
                fldigi.main.rx()
            except Exception:
                pass

    metadata_path = out_root / "raw_wav_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("\nAll requested modes processed.")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
