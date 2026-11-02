#!/usr/bin/env python3
"""CR-DY: Playwright JSON run vs baseline: per test title (file + title) status, branch-only failures, moved messages."""
import json, re, sys
def load(p):
    out = {}
    def walk(s, f):
        f = s.get('file', f)
        for sp in s.get('specs', []):
            for t in sp['tests']:
                rs = t['results']; st = rs[-1]['status'] if rs else 'none'
                msg = (rs[-1].get('error') or {}).get('message', '') if rs else ''
                out[(f.split('/')[-1], sp['title'])] = (st, re.sub(r'\x1b\[[0-9;]*m', '', msg)[:400])
        for c in s.get('suites', []): walk(c, f)
    for s in json.load(open(p))['suites']: walk(s, '')
    return out
b, n = load(sys.argv[1]), load(sys.argv[2])
cnt = lambda d: {k: sum(1 for v in d.values() if v[0] == k) for k in sorted({v[0] for v in d.values()})}
print('baseline', cnt(b)); print('branch', cnt(n))
bad = ('failed', 'timedOut')
print('--- failing on the branch, not failing on baseline')
for k, v in sorted(n.items()):
    if v[0] in bad and (k not in b or b[k][0] not in bad):
        print(k[0], '|', k[1][:110], '|', v[0], '| base', b.get(k, ('new',))[0]); print('   ', v[1].split('\n')[0][:200])
print('--- fixed on the branch'); 
for k, v in sorted(n.items()):
    if v[0] == 'passed' and k in b and b[k][0] in bad: print(k[0], '|', k[1][:110])
print('--- only on baseline'); 
for k in sorted(set(b) - set(n)): print(k[0], '|', k[1][:110])
