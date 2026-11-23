from __future__ import annotations

import json
import sys

import httpx

from riemann.adapters.opgg import (
    OPGG_MODE_URLS,
    OPGG_SCHEMA_VERSION,
    OPGG_USER_AGENT,
    extract_next_payloads_from_html,
    parse_opgg_page_props,
)
from riemann.models import GameMode


def main() -> int:
    headers = {"User-Agent": OPGG_USER_AGENT, "Accept": "text/html,application/xhtml+xml"}
    report: dict[str, dict] = {}
    with httpx.Client(timeout=15.0, headers=headers, follow_redirects=True) as client:
        for mode_str, url in OPGG_MODE_URLS.items():
            mode: GameMode = mode_str  # type: ignore[assignment]
            entry: dict = {"url": url, "schema": OPGG_SCHEMA_VERSION}
            try:
                resp = client.get(url)
                entry["http_status"] = resp.status_code
                resp.raise_for_status()
                props = extract_next_payloads_from_html(resp.text)
                entry["payload_keys"] = sorted(list(props.keys()))[:20]
                obs, rejected = parse_opgg_page_props(props, mode=mode)
                countable = [o for o in obs if o.games is not None]
                tier_only = [o for o in obs if o.games is None and o.curated_tier is not None]
                entry["countable_rows"] = len(countable)
                entry["tier_only_rows"] = len(tier_only)
                entry["rejected_rows"] = rejected
                entry["sample_countable"] = (
                    countable[0].model_dump() if countable else None
                )
            except Exception as exc:
                entry["error"] = str(exc)
            report[mode] = entry

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
