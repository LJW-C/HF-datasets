"""Pure metadata checks shared by online capture and offline generation."""
import re


def bandwidth_reference(modem, readback_bandwidth):
    match=re.match(r'^(?:BPSK|QPSK|8PSK)(\d+)',modem)
    if match:return float(match.group(1))
    match=re.match(r'^OFDM(\d+)',modem)
    if match:return float(match.group(1))
    match=re.match(r'^(?:OLIVIA|Cont)-\d+/(\d+)(K?)$',modem)
    if match:return float(match.group(1))*(1000 if match.group(2) else 1)
    match=re.match(r'^MT63-(\d+)(K?)[SL]$',modem)
    if match:return float(match.group(1))*(1000 if match.group(2) else 1)
    match=re.match(r'^PSK(\d+)C(\d+)$',modem)
    if match:return float(match.group(1))*float(match.group(2))
    return {'DOMEX8':250.,'MFSK16':316.,'MFSK32':500.,'RTTY':200.,'THOR16':300.,'THROB2':72.,'CW':100.}.get(modem,float(readback_bandwidth))
