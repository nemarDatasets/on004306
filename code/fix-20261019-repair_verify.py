"""Compute-node-only, byte-preserving deposit repair and independent reader proof."""
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import traceback
import warnings

import numpy as np

ROOT = Path('/expanse/lustre/projects/sdp190/bpinto/fix-staging')
HERE = ROOT / 'signal-r7'
CACHE = Path('/expanse/lustre/projects/sdp190/bpinto/eegdash-mirror/batch4/data')
assert os.environ.get('SLURM_JOB_ID'), 'must run on compute node'
rows = json.loads((HERE / 'input.json').read_text())


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def repair(row):
    ds, rel = row['dataset'], row['rel']
    root = ROOT / ds
    dest = root / rel
    cached = CACHE / ds / rel
    if ds == 'on004306':
        from scipy.io import loadmat
        # Only the small .set metadata, never an FDT, is read into RAM.
        if dest.is_symlink():
            dest.unlink()
        if not dest.exists():
            shutil.copy2(cached, dest)
        m = loadmat(dest, squeeze_me=True, struct_as_record=False)
        e = m.get('EEG')
        get = (lambda k: getattr(e, k)) if e is not None else (lambda k: m[k])
        size = int(get('nbchan')) * int(get('pnts')) * int(get('trials')) * 4
        assert size == row['expected_size']
        target = root / row['destination']
        if target.is_symlink():
            target.unlink()
        if not target.exists():
            part = target.with_suffix('.fdt.partial')
            url = 'https://s3.amazonaws.com/openneuro.org/ds004306/' + row['source']
            subprocess.run(['curl', '--fail', '--location', '--retry', '3', '--silent', '--show-error', '-o', str(part), url], check=True)
            assert part.stat().st_size == size
            part.rename(target)
        assert target.stat().st_size == size
        sha = digest(target)
        assert row['original_annex_key'] == f'SHA256E-s{size}--{sha}.fdt'
        return dict(row, size=size, sha256=sha, annex_key=row['original_annex_key'], staged_path=str(target), signal_bytes_unchanged=True)
    # Reuse the proven restore-scripts/probe_restore.py header+offset algorithm.
    with cached.open('rb') as f:
        header = bytearray(f.read(5376))
    old = int(header[184:192]); ns = int(header[252:256]); nrec = int(header[236:244]); duration = float(header[244:252])
    assert old in (4352, 4608) and ns == 20
    offset = 256 + ns * (16 + 80 + 8 + 8 + 8 + 8 + 8 + 80)
    spr = [int(header[offset+i*8:offset+(i+1)*8]) for i in range(ns)]
    assert spr == [500] * 20 and duration == 1
    assert header[256:272].strip() == b'Fz'
    assert cached.stat().st_size == old + nrec * sum(spr) * 3
    lost = (5376 - old + 2) // 3
    header[184:192] = b'5376    '
    if dest.is_symlink():
        dest.unlink()
    if not dest.exists():
        with cached.open('rb') as f, dest.open('xb') as g:
            f.seek(old)
            g.write(header)
            shutil.copyfileobj(f, g, 8 * 1024 * 1024)
    # Byte-for-byte comparison of all existing payload, including explicitly BAD bytes.
    with cached.open('rb') as f, dest.open('rb') as g:
        assert g.read(5376) == header
        f.seek(old)
        while True:
            a = f.read(8 * 1024 * 1024); b = g.read(len(a))
            assert a == b
            if not a:
                assert not g.read(1)
                break
    events = dest.with_name(dest.stem.removesuffix('_eeg') + '_events.tsv')
    with events.open(newline='') as f:
        reader = csv.DictReader(f, delimiter='\t'); fields = list(reader.fieldnames); values = list(reader)
    # Existing event values/timing are retained verbatim; do not guess their units.
    if 'description' in fields:
        fields[fields.index('description')] = 'integrity_note'
        for value in values:
            value['integrity_note'] = value.pop('description')
    for field in ('trial_type', 'channel', 'integrity_note', 'integrity_interval_units'):
        if field not in fields:
            fields.append(field)
    for value in values:
        for field in fields:
            value.setdefault(field, 'n/a')
        if value['trial_type'] == 'n/a':
            value['trial_type'] = value.get('event_name', 'n/a')
    label = 'BAD_Fz_header_overlap'
    if not any(v.get('trial_type') == label for v in values):
        event = dict.fromkeys(fields, 'n/a')
        event.update(onset='0', duration=f'{lost/500:.3f}', trial_type=label, channel='Fz', integrity_note=f'First {lost} Fz samples were overwritten by header bytes in the source deposit; not reconstructed. Other channels intact.')
        if 'event_name' in fields:
            event['event_name'] = label
        values.insert(0, event)
    for value in values:
        value['integrity_interval_units'] = 's' if value['trial_type'] == label else 'n/a'
    with events.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter='\t', lineterminator='\n'); writer.writeheader(); writer.writerows(values)
    event_json = events.with_suffix('.json')
    meta = json.loads(event_json.read_text()) if event_json.exists() else {}
    meta.pop('description', None)
    meta.update(trial_type={'Description': 'Original event_name, or BAD_Fz_header_overlap for unrecoverable source bytes.'}, channel={'Description': 'Channel affected by the BAD interval; n/a for original events.'}, integrity_note={'Description': 'Data integrity explanation; no lost samples are synthesized.'})
    meta['integrity_interval_units'] = {'Description': 'Units for the added BAD integrity interval only: s (seconds). Original trigger onset values and their original sample-unit metadata are preserved; they are not silently reinterpreted or corrected by this repair.'}
    event_json.write_text(json.dumps(meta, indent=2)+'\n')
    size = dest.stat().st_size; sha = digest(dest)
    return dict(row, size=size, sha256=sha, annex_key=f'SHA256E-s{size}--{sha}.bdf', staged_path=str(dest), lost_samples=lost, bad_duration_s=lost/500, original_header_field=old, signal_bytes_unchanged=True)


def verify(row):
    import mne
    import mne_bids
    from eegdash import EEGDashDataset
    root = ROOT / row['dataset']; path = root / row['rel']
    out = dict(dataset=row['dataset'], rel=row['rel'], hostname=socket.gethostname(), slurm_job=os.environ['SLURM_JOB_ID'], mne=mne.__version__, mne_bids_version=mne_bids.__version__)
    bp = mne_bids.get_bids_path_from_fname(path, check=False)
    for reader in ('mne_bids', 'eegdash_local'):
        try:
            with warnings.catch_warnings(record=True) as warns:
                warnings.simplefilter('always')
                if reader == 'mne_bids':
                    raw = mne_bids.read_raw_bids(bp, verbose='error')
                else:
                    ds = EEGDashDataset(dataset=row['dataset'], cache_dir=ROOT, download=False, subject=bp.subject, session=bp.session, task=bp.task, run=bp.run, n_jobs=1)
                    assert len(ds.datasets) == 1
                    assert ds.datasets[0].record['storage']['backend'] == 'local'
                    raw = ds.datasets[0].raw
                sf = float(raw.info['sfreq']); n = int(raw.n_times); k = min(int(sf), n)
                first = raw.get_data(start=0, stop=k); last = raw.get_data(start=n-k, stop=n)
                assert np.isfinite(first).all() and np.isfinite(last).all()
                seconds = row.get('seconds', row.get('declared_s'))
                assert abs(n/sf-seconds) < 1/sf
                bad = [(float(a['onset']), float(a['duration']), a['description']) for a in raw.annotations if str(a['description']).startswith('BAD_Fz')]
                if row['dataset'] == 'on003343':
                    assert len(bad) == 1 and bad[0][0] == 0 and abs(bad[0][1]-row['bad_duration_s']) < 1e-6
                out[reader] = dict(ok=True, n_times=n, sfreq=sf, nchan=int(raw.info['nchan']), seconds=n/sf, first_last_finite=True, bad=bad, warnings=[str(w.message)[:500] for w in warns])
        except Exception as exc:
            out[reader] = dict(ok=False, error=repr(exc), traceback=traceback.format_exc())
    if row['dataset'] == 'on004306':
        # EEGDash normalizes datfile in place. Restore original SET metadata so
        # the retained staging tree matches the pointer-only local branch.
        shutil.copy2(CACHE / row['dataset'] / row['rel'], path)
        out['original_set_restored_after_reader_normalization'] = True
    return out


if __name__ == '__main__':
    selected = [r for r in rows if len(sys.argv) < 2 or r['dataset'] == sys.argv[1]]
    for row in selected:
        try:
            result = repair(row)
            with (HERE / (row['dataset'] + '-manifest.jsonl')).open('a') as f:
                f.write(json.dumps(result)+'\n')
            proof = verify(result)
        except Exception as exc:
            proof = dict(dataset=row['dataset'], rel=row['rel'], repair_error=repr(exc), traceback=traceback.format_exc())
        with (HERE / (row['dataset'] + '-verification.jsonl')).open('a') as f:
            f.write(json.dumps(proof)+'\n')
        print(json.dumps(proof), flush=True)
