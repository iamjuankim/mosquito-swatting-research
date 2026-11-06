"""Read-only verification of the explicit public payload; no simulation imports."""
from pathlib import Path
import ast
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[1]

def main():
    files = json.loads((ROOT / 'provenance/release_allowlist.json').read_text(encoding='utf-8'))
    sizes = []
    for rel in files:
        p = ROOT / rel
        assert p.resolve().is_relative_to(ROOT) and p.is_file(), rel
        raw = p.read_bytes(); text = raw.decode('utf-8-sig')
        assert len(raw) < 10_000_000, rel
        # Split detector literals to avoid matching this scanner itself.
        assert not re.search(r'(?i)[a-z]:[/\\]+us' + r'ers[/\\]+|/ho' + r'me/[^/]+/', text), rel
        for pattern in ['gh' + r'p_[A-Za-z0-9]{30,}', 'github_' + r'pat_[A-Za-z0-9_]{30,}',
                        '-----BEGIN ' + r'(?:RSA |OPENSSH |EC )?PRIVATE KEY-----']:
            assert not re.search(pattern, text), rel
        if p.suffix == '.py': ast.parse(text, filename=rel)
        sizes.append(len(raw))
    lineage = json.loads((ROOT / 'provenance/source_lineage.json').read_text(encoding='utf-8'))
    lookup = {r['release']:r for r in lineage}
    for r in lineage:
        assert hashlib.sha256((ROOT / r['release']).read_bytes()).hexdigest() == r['release_sha256'], r['release']
    for rel, expected in json.loads((ROOT / 'provenance/payload_hashes.json').read_text(encoding='utf-8')).items():
        raw = (ROOT / rel).read_bytes()
        assert len(raw) == expected['bytes'] and hashlib.sha256(raw).hexdigest() == expected['sha256'], rel
    matches = []; mismatches = []
    manifests = [('mq10_audit_stage7_h2_stable_20261105', 'code_manifest.json', 'code_sha256'),
                 ('mq10_audit_stage9g_p1p2_20261106/run_04', 'run_code_manifest.json', None),
                 ('mq10_audit_stage9h_p3_20261106', 'p3_run_code_manifest.json', None)]
    for folder, name, key in manifests:
        m = json.loads((ROOT / 'archive' / folder / name).read_text(encoding='utf-8-sig'))
        if key: m = m[key]
        for rel, digest in m.items():
            release = 'archive/' + folder + '/' + rel.replace('\\', '/')
            assert release in lookup, 'Manifest code absent: ' + release
            entry = dict(file=release, matches=lookup[release]['source_sha256'] == digest)
            (matches if entry['matches'] else mismatches).append(entry)
    assert not mismatches, mismatches
    print(json.dumps(dict(public_files=len(files), public_bytes=sum(sizes),
                          historical_code_hash_matches=len(matches), mismatches=mismatches,
                          scope='Payload/static/source identity only; no historical Gate rerun')))

if __name__ == '__main__':
    main()
