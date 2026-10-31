"""Record Fldigi sources with readback, saved payloads and spectral checks."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import re
import sys
import time
import xmlrpc.client

sys.path.insert(0, str(Path(__file__).parent / 'python_packages'))
repository = Path(__file__).resolve().parents[1]
if not (repository / 'generate_fldigi_wavs.py').is_file():
    repository = repository / 'hf-digital-mode-iq-dataset-github'
sys.path.insert(0, str(repository))
import numpy as np
from scipy import signal
from generate_fldigi_wavs import DEFAULT_MODES, make_random_payload, import_audio_packages
from source_checks import bandwidth_reference


class TimeoutTransport(xmlrpc.client.Transport):
    def make_connection(self, host):
        connection = super().make_connection(host)
        connection.timeout = 30
        return connection


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def wait_rx(rpc):
    rpc.main.abort()
    rpc.main.rx()
    for _ in range(50):
        if str(rpc.main.get_trx_state()).lower() == 'rx':
            return
        time.sleep(.1)
    raise RuntimeError('Fldigi failed to return to RX')


def configure(rpc, modem, carrier):
    wait_rx(rpc)
    rpc.modem.set_by_name(modem)
    time.sleep(.8)
    selection_method='set_by_name'
    if rpc.modem.get_name()!=modem:
        names=rpc.modem.get_names()
        if modem not in names:raise ValueError('Requested modem not present')
        rpc.modem.set_by_id(names.index(modem))
        time.sleep(.8)
        selection_method='set_by_id_after_name_readback_mismatch'
    for option in ['txid', 'rsid', 'afc', 'reverse']:
        getattr(rpc.main, 'set_' + option)(False)
        if getattr(rpc.main, 'get_' + option)():
            raise RuntimeError(option + ' did not disable')
    rpc.modem.set_carrier(int(carrier))
    time.sleep(.2)
    state = dict(actual_modem=rpc.modem.get_name(), actual_id=rpc.modem.get_id(),
                 carrier_hz=rpc.modem.get_carrier(), bandwidth_hz=rpc.modem.get_bandwidth(),
                 txid=rpc.main.get_txid(), rsid=rpc.main.get_rsid(),
                 afc=rpc.main.get_afc(), reverse=rpc.main.get_reverse())
    state['selection_method']=selection_method
    if state['actual_modem'] != modem or abs(state['carrier_hz'] - carrier) > 1:
        raise RuntimeError('Requested state differs from readback: ' + str(state))
    return state


def spectral_qc(audio, fs, carrier, nominal_bandwidth):
    x = np.asarray(audio).reshape(-1)
    interior = x[int(3 * fs):-fs]
    if interior.size < fs:
        raise ValueError('Recording is too short')
    f,p = signal.welch(interior-interior.mean(), fs, nperseg=8192)
    c = np.cumsum(p) / max(float(p.sum()), 1e-30)
    lo, hi = np.interp([.005,.995], c, f)
    power = float(np.mean(interior**2))
    clipped = float(np.mean(np.abs(interior) >= .999))
    bw = float(hi-lo)
    reasons=[]
    if power < 1e-7: reasons.append('near_silent')
    if clipped > .001: reasons.append('clipping')
    if nominal_bandwidth > 0 and not .45*nominal_bandwidth <= bw <= 2.2*nominal_bandwidth:
        reasons.append('occupied_bandwidth_mismatch')
    if lo < 25 or hi > fs/2-100: reasons.append('audio_band_edge')
    return dict(obw99_hz=bw, audio_low99_hz=float(lo), audio_high99_hz=float(hi),
                baseband_low99_hz=float(lo-carrier), baseband_high99_hz=float(hi-carrier),
                interior_power=power, clipped_fraction=clipped, reasons=reasons,
                passed=not reasons, scope='Control readback and energy/spectrum QC; no independent modem decoding.')


def record(rpc, mode, args, index):
    folder = args.out_root / mode['folder']
    folder.mkdir(parents=True, exist_ok=True)
    metadata_path=folder/'recording.json'
    if metadata_path.exists():
        old=json.loads(metadata_path.read_text(encoding='utf-8'))
        if old.get('qc',{}).get('passed') and (folder/'source.wav').exists():
            print('SKIP VERIFIED',mode['folder'],flush=True)
            return old
        raise FileExistsError('Existing rejected/incomplete recording: '+str(folder))
    carrier=2200 if mode['fldigi_modem']=='OFDM3500' else 1500
    state=configure(rpc,mode['fldigi_modem'],carrier)
    seed=args.seed+index
    random.seed(seed)
    payload=make_random_payload(30000)
    payload_path=folder/'payload.txt'
    payload_path.write_text(payload,encoding='utf-8')
    print('  queue payload',flush=True)
    rpc.text.clear_tx()
    rpc.text.add_tx(payload[:1024])
    queued_chars=1024
    print('  start audio capture',flush=True)
    data=sd.rec(int(args.duration_sec*args.record_fs),samplerate=args.record_fs,channels=1,
                dtype='float32',device=args.device_id)
    monitored=[]
    try:
        time.sleep(.25)
        print('  start TX',flush=True)
        rpc.main.tx()
        print('  TX acknowledged',flush=True)
        deadline=time.monotonic()+args.duration_sec-.25
        while time.monotonic()<deadline:
            time.sleep(min(2,max(.01,deadline-time.monotonic())))
            name=rpc.modem.get_name()
            txid=rpc.main.get_txid()
            carrier_readback=rpc.modem.get_carrier()
            trx=rpc.main.get_trx_state()
            monitored.append(dict(modem=name,txid=txid,carrier_hz=carrier_readback,trx=trx))
            if name!=state['actual_modem'] or txid or abs(carrier_readback-carrier)>1:
                raise RuntimeError('Transmitter state changed during recording')
            if str(trx).lower()!='tx':
                raise RuntimeError('Fldigi left TX before capture finished')
            if queued_chars<len(payload):
                rpc.text.add_tx(payload[queued_chars:queued_chars+512])
                queued_chars+=512
        sd.wait()
    finally:
        sd.stop()
        wait_rx(rpc)
    reference=bandwidth_reference(mode['fldigi_modem'],state['bandwidth_hz'])
    qc=spectral_qc(data,args.record_fs,carrier,reference)
    qc['bandwidth_screening_reference_hz']=reference
    path=folder/('source.wav' if qc['passed'] else 'rejected_source.wav')
    sf.write(path,data,args.record_fs,subtype='PCM_24')
    result=dict(mode_name=mode['folder'],requested_modem=mode['fldigi_modem'],readback=state,
                fldigi_version=rpc.fldigi.version(),wav=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                payload_file=str(payload_path.resolve()),payload_sha256=hashlib.sha256(payload_path.read_bytes()).hexdigest(),
                payload_seed=seed,payload_queued_chars=queued_chars,record_fs=args.record_fs,duration_sec=args.duration_sec,channels=1,subtype='PCM_24',
                device=sd.query_devices(args.device_id),monitor=monitored,qc=qc,
                payload_interval_seconds=[3,args.duration_sec-1],
                rsid_verification='TX/RX RSID disabled with readback before TX; TX RSID monitored during TX. No waveform-level RSID decoder.')
    save_json(metadata_path,result)
    print(mode['folder'], 'PASS' if qc['passed'] else 'REJECT', qc,flush=True)
    return result


def main():
    global sd, sf
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out-root',type=Path,default=Path(__file__).parent/'raw_verified')
    p.add_argument('--duration-sec',type=int,default=60)
    p.add_argument('--record-fs',type=int,default=16000)
    p.add_argument('--device-id',type=int,help='Actual CABLE Output input device ID on this machine')
    p.add_argument('--list-devices',action='store_true')
    p.add_argument('--seed',type=int,default=20261029)
    p.add_argument('--modes',default='benchmark')
    p.add_argument('--xmlrpc-url',default='http://127.0.0.1:7362')
    args=p.parse_args()
    sd, sf = import_audio_packages()
    if args.list_devices:
        print(sd.query_devices())
        return
    if args.device_id is None: p.error('--device-id is required; inspect --list-devices first')
    if args.duration_sec < 10: p.error('duration must be at least 10 seconds')
    modes=DEFAULT_MODES.copy()
    if args.modes!='benchmark':
        wanted=args.modes.split(',')
        modes=[m for m in DEFAULT_MODES if m['folder'] in wanted or m['fldigi_modem'] in wanted]
        found={m['folder'] for m in modes}|{m['fldigi_modem'] for m in modes}
        for name in wanted:
            if name not in found:
                modes.append(dict(folder=re.sub('[^A-Za-z0-9]+','_',name),fldigi_modem=name))
    args.out_root.mkdir(parents=True,exist_ok=True)
    rpc=xmlrpc.client.ServerProxy(args.xmlrpc_url,allow_none=True,transport=TimeoutTransport())
    available=rpc.modem.get_names()
    device=sd.query_devices(args.device_id)
    if 'CABLE Output' not in device['name'] or device['max_input_channels'] < 1:
        raise RuntimeError('Recording device is not the virtual cable output: '+str(device))
    sd.check_input_settings(device=args.device_id,channels=1,samplerate=args.record_fs)
    manifest=dict(status='recording',recordings=[],failures=[])
    for index,mode in enumerate(modes):
        print('RECORD',index+1,len(modes),mode,flush=True)
        try:
            if mode['fldigi_modem'] not in available: raise ValueError('Modem is not available')
            manifest['recordings'].append(record(rpc,mode,args,index))
        except Exception as exc:
            manifest['failures'].append(dict(mode=mode,error=str(exc)))
            print('FAILED',mode,repr(exc),flush=True)
            try: wait_rx(rpc)
            except Exception: pass
        save_json(args.out_root/'recordings_manifest.json',manifest)
    manifest['status']='complete' if not manifest['failures'] and all(r['qc']['passed'] for r in manifest['recordings']) else 'needs_review'
    save_json(args.out_root/'recordings_manifest.json',manifest)
    print('FINAL STATUS',manifest['status'],flush=True)
    if manifest['status']!='complete': raise SystemExit(2)


if __name__=='__main__':main()
