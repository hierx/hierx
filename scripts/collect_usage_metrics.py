"""Collect durable usage metrics for HierX 0.1.1.

The script intentionally keeps two PyPI series:

* pypistats.org's mirror-filtered package series (broad, but not versioned)
* ClickPy's version-specific pip/uv series (narrow, conservative)

GitHub views and clones require a token with repository Administration: read.
When that permission is unavailable, public metrics are still collected and the
snapshot records traffic_available=false.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import sys
import urllib.error
import urllib.request
from collections.abc import Iterable
from pathlib import Path
from typing import Any

PACKAGE = "hierx"
PACKAGE_VERSION = "0.1.1"
RELEASE_TAG = "v0.1.1"
DEFAULT_REPOSITORY = "hierx/hierx"
USER_AGENT = "hierx-usage-tracker/1.0 (+https://github.com/hierx/hierx)"


def request_json(
    url: str,
    *,
    token: str | None = None,
    data: bytes | None = None,
    timeout: int = 45,
) -> Any:
    headers = {
        "Accept": "application/vnd.github+json" if "api.github.com" in url else "application/json",
        "User-Agent": USER_AGENT,
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
        headers["X-GitHub-Api-Version"] = "2022-11-28"
    request = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8")
    if url.startswith("https://sql-clickhouse.clickhouse.com/"):
        return [json.loads(line) for line in body.splitlines() if line.strip()]
    return json.loads(body)


def upsert_csv(path: Path, key_fields: tuple[str, ...], rows: Iterable[dict[str, Any]]) -> None:
    incoming = list(rows)
    existing: list[dict[str, str]] = []
    if path.exists():
        with path.open(newline="", encoding="utf-8") as handle:
            existing = list(csv.DictReader(handle))

    fieldnames: list[str] = []
    for row in [*existing, *incoming]:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    if not fieldnames:
        return

    merged: dict[tuple[str, ...], dict[str, Any]] = {}
    for row in [*existing, *incoming]:
        key = tuple(str(row.get(field, "")) for field in key_fields)
        merged[key] = row

    def sort_key(item: tuple[tuple[str, ...], dict[str, Any]]) -> tuple[str, ...]:
        return item[0]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for _, row in sorted(merged.items(), key=sort_key):
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def fetch_pypistats() -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    overall_url = f"https://pypistats.org/api/packages/{PACKAGE}/overall?mirrors=false"
    recent_url = f"https://pypistats.org/api/packages/{PACKAGE}/recent"
    overall = request_json(overall_url)
    recent = None
    try:
        recent = request_json(recent_url)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
        print(f"warning: PyPI recent totals unavailable: {exc}", file=sys.stderr)

    rows = [
        {
            "date": item["date"],
            "downloads": item["downloads"],
            "scope": "all_versions",
            "mirror_policy": item.get("category", "without_mirrors"),
        }
        for item in overall.get("data", [])
        if item.get("category") == "without_mirrors"
    ]
    return rows, recent


def fetch_version_specific_pypi() -> list[dict[str, Any]]:
    query = f"""
        SELECT
            toString(date) AS date,
            sum(count) AS downloads
        FROM pypi.pypi_downloads_per_day_by_version_by_installer_by_type
        WHERE project = '{PACKAGE}'
          AND version = '{PACKAGE_VERSION}'
          AND installer IN ('pip', 'uv')
          AND type IN ('bdist_wheel', 'sdist')
        GROUP BY date
        ORDER BY date
        FORMAT JSONEachRow
    """
    url = "https://sql-clickhouse.clickhouse.com/?user=play"
    result = request_json(url, data=query.encode("utf-8"))
    return [
        {
            "date": item["date"],
            "downloads": int(item["downloads"]),
            "package_version": PACKAGE_VERSION,
            "installers": "pip+uv",
            "file_types": "bdist_wheel+sdist",
        }
        for item in result
    ]


def fetch_github(
    repository: str, token: str | None
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    root = f"https://api.github.com/repos/{repository}"
    repo = request_json(root, token=token)
    releases = request_json(f"{root}/releases", token=token)
    release = next((item for item in releases if item.get("tag_name") == RELEASE_TAG), None)
    release_asset_downloads = sum(
        asset.get("download_count", 0) for asset in (release or {}).get("assets", [])
    )

    traffic: dict[str, Any] = {"available": False, "views": None, "clones": None}
    if token:
        try:
            traffic["views"] = request_json(f"{root}/traffic/views?per=day", token=token)
            traffic["clones"] = request_json(f"{root}/traffic/clones?per=day", token=token)
            traffic["available"] = True
        except urllib.error.HTTPError as exc:
            if exc.code not in (401, 403, 404):
                raise
            print(
                "warning: GitHub traffic unavailable; TRAFFIC_TOKEN needs Administration: read",
                file=sys.stderr,
            )

    public_snapshot = {
        "stars": repo.get("stargazers_count", 0),
        "forks": repo.get("forks_count", 0),
        "subscribers": repo.get("subscribers_count", 0),
        "open_issues": repo.get("open_issues_count", 0),
        "release_asset_downloads": release_asset_downloads,
    }
    return public_snapshot, releases, traffic


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("usage-metrics"))
    args = parser.parse_args()

    repository = os.environ.get("GITHUB_REPOSITORY", DEFAULT_REPOSITORY)
    token = os.environ.get("TRAFFIC_TOKEN") or os.environ.get("GH_TOKEN")
    today = dt.datetime.now(dt.timezone.utc).date().isoformat()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    failures: list[str] = []
    pypi_rows: list[dict[str, Any]] = []
    pypi_recent: dict[str, Any] | None = None
    try:
        pypi_rows, pypi_recent = fetch_pypistats()
        upsert_csv(args.output_dir / "pypi_overall_daily.csv", ("date",), pypi_rows)
    except (
        urllib.error.HTTPError,
        urllib.error.URLError,
        TimeoutError,
        ValueError,
    ) as exc:
        failures.append(f"pypistats: {exc}")

    try:
        version_rows = fetch_version_specific_pypi()
        upsert_csv(args.output_dir / "pypi_0.1.1_pip_uv_daily.csv", ("date",), version_rows)
    except (
        urllib.error.HTTPError,
        urllib.error.URLError,
        TimeoutError,
        ValueError,
    ) as exc:
        failures.append(f"ClickPy: {exc}")

    public: dict[str, Any] = {}
    traffic: dict[str, Any] = {"available": False, "views": None, "clones": None}
    try:
        public, _, traffic = fetch_github(repository, token)
    except (
        urllib.error.HTTPError,
        urllib.error.URLError,
        TimeoutError,
        ValueError,
    ) as exc:
        failures.append(f"GitHub: {exc}")

    if traffic.get("available"):
        upsert_csv(
            args.output_dir / "github_views_daily.csv",
            ("date",),
            (
                {
                    "date": item["timestamp"][:10],
                    "views": item["count"],
                    "unique_visitors": item["uniques"],
                }
                for item in traffic["views"].get("views", [])
            ),
        )
        upsert_csv(
            args.output_dir / "github_clones_daily.csv",
            ("date",),
            (
                {
                    "date": item["timestamp"][:10],
                    "clones": item["count"],
                    "unique_cloners": item["uniques"],
                }
                for item in traffic["clones"].get("clones", [])
            ),
        )

    recent_data = (pypi_recent or {}).get("data", {})
    snapshot = {
        "snapshot_date_utc": today,
        "pypi_last_day": recent_data.get("last_day", ""),
        "pypi_last_week": recent_data.get("last_week", ""),
        "pypi_last_month": recent_data.get("last_month", ""),
        "github_stars": public.get("stars", ""),
        "github_forks": public.get("forks", ""),
        "github_subscribers": public.get("subscribers", ""),
        "github_open_issues": public.get("open_issues", ""),
        "v0.1.1_release_asset_downloads": public.get("release_asset_downloads", ""),
        "traffic_available": str(bool(traffic.get("available"))).lower(),
        "github_views_14d": (traffic.get("views") or {}).get("count", ""),
        "github_unique_visitors_14d": (traffic.get("views") or {}).get("uniques", ""),
        "github_clones_14d": (traffic.get("clones") or {}).get("count", ""),
        "github_unique_cloners_14d": (traffic.get("clones") or {}).get("uniques", ""),
    }
    upsert_csv(args.output_dir / "snapshots.csv", ("snapshot_date_utc",), [snapshot])

    status = {
        "collected_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "repository": repository,
        "package": PACKAGE,
        "package_version": PACKAGE_VERSION,
        "traffic_available": bool(traffic.get("available")),
        "failures": failures,
    }
    (args.output_dir / "status.json").write_text(
        json.dumps(status, indent=2) + "\n", encoding="utf-8"
    )
    for failure in failures:
        print(f"warning: {failure}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
