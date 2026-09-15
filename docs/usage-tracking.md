# HierX 0.1.1 usage tracking

The `Archive usage metrics` workflow runs daily and writes durable CSV series to
the `usage-metrics` branch. It can also be run manually from the Actions tab.

## Metrics and interpretation

| File | Meaning |
| --- | --- |
| `pypi_overall_daily.csv` | Daily downloads of the `hierx` package reported by pypistats.org with known mirrors excluded. The source is not version-specific. Until another HierX release is published, post-2026-03-17 values are attributable to 0.1.1. |
| `pypi_0.1.1_pip_uv_daily.csv` | Conservative version-specific series from ClickPy: only 0.1.1 wheel/source downloads whose installer is identified as `pip` or `uv`. |
| `github_views_daily.csv` | Repository page views and unique visitors. GitHub exposes only the latest 14 days, so daily archiving is necessary. |
| `github_clones_daily.csv` | Full clones and unique cloners. GitHub exposes only the latest 14 days. Ordinary fetches are not counted. |
| `snapshots.csv` | Daily public repository totals, recent PyPI totals, release-asset downloads, and rolling 14-day GitHub traffic totals. |
| `status.json` | Freshness, source availability, and any non-fatal collection errors from the latest run. |

Download figures are signals of attention, not counts of people. CI jobs,
dependency resolvers, mirrors, repeated installations, and automated tools can
all generate downloads. The narrow `pip`/`uv` series is useful as a conservative
comparison, while the pypistats.org series is the broader headline measure.

## One-time GitHub traffic authorization

The normal Actions token cannot read repository traffic. To enable views and
clones, create a fine-grained personal access token restricted to `hierx/hierx`
with **Administration: read-only**, then save it as an Actions repository secret
named `TRAFFIC_TOKEN`.

Without that secret, the workflow still records PyPI downloads, stars, forks,
subscribers, issues, and release-asset downloads. `traffic_available` will be
`false` until the secret is present and authorized for the organization.

## Baseline captured 2026-09-15

| Measure | Value |
| --- | ---: |
| pypistats.org downloads without known mirrors, 2026-03-18 through 2026-09-12 | 197 |
| 0.1.1 downloads identified specifically as `pip`/`uv`, 2026-03-17 through 2026-08-28 | 27 |
| Downloads in the latest pypistats day / week / month | 0 / 1 / 8 |
| GitHub stars / forks / subscribers | 0 / 0 / 0 |
| v0.1.1 GitHub release-asset downloads | 0 (the release has no attached assets) |

The arXiv paper is [arXiv:2609.08676](https://arxiv.org/abs/2609.08676).
arXiv does not expose an author-facing per-paper views/downloads counter, so it
is not included as a numeric series. Citations and indexed mentions should be
tracked separately as scholarly-attention indicators.
