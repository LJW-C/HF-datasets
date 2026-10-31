"""Verify a DOI-bound upload package without asserting public release."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def verify(root,expected_doi):
    records=json.loads((root/'SHA256SUMS.json').read_text())
    for item in records:
        path=root/item['file']
        assert path.stat().st_size==item['bytes'] and sha(path)==item['sha256'],path.name
        if path.suffix=='.zip':
            with zipfile.ZipFile(path) as z:
                assert z.testzip() is None,path.name
                assert len(z.namelist())==len(set(z.namelist())),path.name
            print('VERIFIED',path.name,flush=True)
    manifest=json.loads((root/'dataset_manifest.json').read_text())
    assert manifest['revised_version_doi']==expected_doi
    assert manifest['status']=='DOI_RESERVED_PUBLICATION_PENDING'
    assert manifest['published_at_utc'] is None
    assert manifest['creators']==[{'name':'Junwei Li','family_name':'Li','given_name':'Junwei'}]
    assert manifest['public_code_commit'] is None
    with zipfile.ZipFile(root/'benchmark_17_2.0.0-rc1.zip') as z:
        with z.open('X.npy') as stream:
            assert np.lib.format.read_magic(stream)==(1,0)
            shape,fortran,dtype=np.lib.format.read_array_header_1_0(stream)
        assert shape==(163200,2048) and not fortran and str(dtype)=='complex64'
    with (root/'source_wav_metadata.csv').open(encoding='utf-8-sig') as f:
        sources=list(csv.DictReader(f))
    assert len(sources)==33 and sum(s['revised_class_id']!='' for s in sources)==17
    with zipfile.ZipFile(root/'screened_sources_33_2.0.0-rc1.zip') as z:
        assert sum(n.endswith('/source.wav') for n in z.namelist())==33
        for r in sources:
            assert hashlib.sha256(z.read(r['source_wav_path'])).hexdigest()==r['sha256']
    assert sha(root/'code_snapshot_2.0.0-rc1.zip')==manifest['code_snapshot_sha256']
    with zipfile.ZipFile(root/'code_snapshot_2.0.0-rc1.zip') as z:
        for r in manifest['source_code_files']:
            assert hashlib.sha256(z.read(r['path'])).hexdigest()==r['sha256']
    report=dict(status='passed',reserved_doi=expected_doi,public_release_verified=False,
        upload_files=len(records)+1,source_wavs=33,benchmark_classes=17,shape=list(shape))
    print(json.dumps(report,indent=2),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--expected-doi',default='10.5281/zenodo.23795076')
    a=p.parse_args()
    verify(a.root,a.expected_doi)
