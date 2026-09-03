# Dependency security notes

## Why `mlflow` is pinned to latest, not an older "LTS" release

Tried pinning back to `mlflow==3.2.0` — the newest release before both of the
two issues `pip-audit` flags on 3.15.2 appear (bisected against real PyPI
wheels on 2026-09-03):

| mlflow version | `cryptography` pin | Vulnerable AI-Gateway code (CVE-2026-71211) |
|---|---|---|
| 3.0.0 – 3.2.0 | none (direct) | absent |
| 3.3.0 – 3.5.0 | `<46` / `<47` | absent |
| 3.8.0 – 3.14.0 | `<47` / `<49` | **present** |
| 3.15.0 – 3.15.2 | `<50` | **present** |

3.2.0 looks clean on both counts. It isn't a net win: `pip-audit` against a
fresh `mlflow==3.2.0` install found **31 vulnerabilities across mlflow and
pyarrow** — CVEs that 3.15.2 already carries fixes for (PYSEC-2026-94/93/2655/
195/424/421/425/423/2220/1639/1656/2219/2221/2222/2654/2656/2657/2660/2659/
2661/2658/3687/3686, GHSA-gqvg-gmmx-x4hm, plus PYSEC-2026-113 in the older
pyarrow it pulls in). Trading one unfixed-but-inapplicable CVE for thirty
already-patched ones is a worse position, not a better one.

**Conclusion: stay on latest mlflow.** For actively-maintained software, the
newest release is usually the most secure release, not the least — the
"pin to something older and stabler" instinct only pays off when the
vulnerability was introduced recently by the newest version specifically,
and even then only if nothing else regresses. Here it wasn't a good trade in
either direction. The two remaining findings are accepted as documented,
inapplicable risk (see `.github/workflows/ci.yml`'s `pip-audit` step for the
per-CVE reasoning) rather than downgraded around.

Re-run this bisection if either CVE ever gets a real fix, or before any
future dependency-security discussion defaults to "just pin older."
