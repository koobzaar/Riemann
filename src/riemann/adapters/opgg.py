from __future__ import annotations

import json
import re
import time
from typing import Any, Iterator

import httpx
from bs4 import BeautifulSoup

from riemann.adapters.cdragon import canonicalize_slug
from riemann.models import AdapterPayload, GameMode, Observation
from riemann.stats_engine import StatsValidationError, recover_unique_wins_from_rate

OPGG_SCHEMA_VERSION = "opgg_schema_v1"
OPGG_USER_AGENT = (
    "RiemannLocalStatsClient/0.1 (+https://github.com/local/riemann; polite-sync)"
)

OPGG_MODE_URLS: dict[GameMode, str] = {
    "arena": "https://op.gg/lol/modes/arena",
    "aram_mayhem": "https://op.gg/lol/modes/aram-mayhem",
}

_RSC_PUSH_RE = re.compile(r'self\.__next_f\.push\(\[1,"(.*?)"\]\)', re.DOTALL)


def _walk_dicts(obj: Any) -> Iterator[dict[str, Any]]:
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from _walk_dicts(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_dicts(v)


def _decode_rsc_dicts(rsc_raw: str) -> list[dict[str, Any]]:
    dec = json.JSONDecoder()
    out: list[dict[str, Any]] = []
    pos = 0
    n = len(rsc_raw)
    while pos < n:
        idx = rsc_raw.find("{", pos)
        if idx == -1:
            break
        try:
            obj, end = dec.raw_decode(rsc_raw, idx)
            out.extend(_walk_dicts(obj))
            pos = end
        except Exception:
            pos = idx + 1
    return out


def _normalize_rsc_to_page_props(rsc_raw: str) -> dict[str, Any]:
    """Extract normalized opgg_schema_v1 pageProps from Next.js App Router RSC chunks."""
    dicts = _decode_rsc_dicts(rsc_raw)
    patch: str | None = None
    versions: list[str] = []
    mode_from_rsc: str | None = None
    champion: str | None = None
    augments: list[dict[str, Any]] = []
    items: list[dict[str, Any]] = []
    champion_tiers: list[dict[str, Any]] = []
    child_entity_dict_ids: set[int] = set()

    # Pass 1: collect metadata, entity arrays, and child dict ids to exclude from baseline lookup
    for d in dicts:
        if patch is None and isinstance(d.get("query"), dict) and d["query"].get("patch"):
            patch = str(d["query"]["patch"])
        if not versions and isinstance(d.get("versions"), list) and d["versions"]:
            versions = [str(v) for v in d["versions"]]
        if isinstance(d.get("pathParams"), dict):
            pmode = d["pathParams"].get("mode")
            if mode_from_rsc is None and isinstance(pmode, str) and pmode:
                mode_from_rsc = "aram_mayhem" if "mayhem" in pmode.lower() else ("arena" if "arena" in pmode.lower() else None)
            if champion is None:
                slug = d["pathParams"].get("slug")
                if isinstance(slug, list) and slug:
                    champion = str(slug[0])
                elif isinstance(slug, str) and slug and slug != "$undefined":
                    champion = slug

        if isinstance(d.get("augments"), list) and d["augments"] and isinstance(d["augments"][0], dict):
            for row in d["augments"]:
                if isinstance(row, dict):
                    augments.append(row)
                    child_entity_dict_ids.add(id(row))
        if isinstance(d.get("items"), list) and d["items"] and isinstance(d["items"][0], dict):
            if "short_title" not in d["items"][0]:
                for row in d["items"]:
                    if isinstance(row, dict):
                        items.append(row)
                        child_entity_dict_ids.add(id(row))
        if isinstance(d.get("synergies"), list):
            for row in d["synergies"]:
                if isinstance(row, dict):
                    child_entity_dict_ids.add(id(row))
        if isinstance(d.get("champions"), list) and d["champions"] and isinstance(d["champions"][0], dict):
            first = d["champions"][0]
            if "key" in first and ("tier" in first or "win_rate" in first):
                for row in d["champions"]:
                    if isinstance(row, dict):
                        champion_tiers.append(row)
                        child_entity_dict_ids.add(id(row))

    # Pass 2: extract target champion baseline_rate ONLY from target champion summary or matching champion_tiers entry
    target_champ_slug = canonicalize_slug(champion) if champion else None
    baseline_rate: float | None = None

    def _to_prop(v: float) -> float:
        return v / 100.0 if v > 1.0 else v

    if target_champ_slug:
        for row in champion_tiers:
            rkey = canonicalize_slug(str(row.get("key") or row.get("name") or ""))
            wr = row.get("win_rate")
            if rkey == target_champ_slug and isinstance(wr, (int, float)) and not isinstance(wr, bool):
                baseline_rate = _to_prop(float(wr))
                break

    if baseline_rate is None:
        for d in dicts:
            if id(d) in child_entity_dict_ids:
                continue
            wr = d.get("win_rate")
            if (
                "pick_rate" in d
                and isinstance(wr, (int, float))
                and not isinstance(wr, bool)
            ):
                # Reject dicts that belong to a different named champion or entity
                d_key = d.get("key") or d.get("champion_key") or d.get("canonical_id")
                if d_key is not None and target_champ_slug and canonicalize_slug(str(d_key)) != target_champ_slug:
                    continue
                if "name" in d and "tier" in d and target_champ_slug and canonicalize_slug(str(d["name"])) != target_champ_slug:
                    continue
                baseline_rate = _to_prop(float(wr))
                break

    out: dict[str, Any] = {
        "patch": patch or (versions[0] if versions else "unknown"),
        "versions": versions,
        "champion": champion or "all",
        "champion_baseline_rate": baseline_rate,
        "augments": augments,
        "items": items,
        "champion_tiers": champion_tiers,
        "rsc_parsed": True,
    }
    if mode_from_rsc is not None:
        out["mode"] = mode_from_rsc
    return out


def extract_next_payloads_from_html(html: str) -> dict[str, Any]:
    """Extract SSR JSON from `<script id="__NEXT_DATA__">` or Next.js App Router RSC payloads."""
    soup = BeautifulSoup(html, "html.parser")
    script = soup.find("script", id="__NEXT_DATA__")
    if script is not None and script.string:
        data = json.loads(script.string)
        return data.get("props", {}).get("pageProps", data)

    chunks: list[str] = []
    for match in _RSC_PUSH_RE.finditer(html):
        try:
            chunks.append(json.loads(f'"{match.group(1)}"'))
        except Exception:
            continue
    if chunks:
        return _normalize_rsc_to_page_props("\n".join(chunks))
    raise ValueError("No supported OP.GG __NEXT_DATA__ or RSC payload found in HTML")


def parse_opgg_page_props(
    page_props: dict[str, Any],
    *,
    mode: GameMode,
    source_id: str = "opgg",
    overlap_group: str = "riot_global_pool",
    source_priority: int = 30,
) -> tuple[list[Observation], int]:
    """Parse OP.GG `opgg_schema_v1` pageProps (from __NEXT_DATA__ or RSC) into Observations.

    Returns (observations, rejected_rows_count).
    Tier-only rows are stored with `curated_tier` and `games=None` (never fabricated into wins).
    """
    patch = str(page_props.get("patch") or page_props.get("version") or "unknown")
    champion_id = canonicalize_slug(str(page_props.get("champion") or "unknown"))
    baseline_wins = page_props.get("champion_baseline_wins")
    baseline_games = page_props.get("champion_baseline_games")
    baseline_rate = page_props.get("champion_baseline_rate")
    default_metric = "win" if mode == "aram_mayhem" else "top4"

    observations: list[Observation] = []
    seen_pks: set[tuple[str, ...]] = set()
    rejected = 0

    def _append_unique(obs: Observation) -> None:
        pk = obs.primary_key()
        if pk not in seen_pks:
            seen_pks.add(pk)
            observations.append(obs)

    def _parse_group(rows: list[dict[str, Any]], entity_type: str) -> None:
        nonlocal rejected
        for row in rows:
            eid = canonicalize_slug(str(row.get("canonical_id") or row.get("name") or row.get("key") or ""))
            if not eid:
                rejected += 1
                continue
            slot_key = str(row.get("slot_key") or "all")
            metric = str(row.get("metric_type") or default_metric)
            tier = row.get("tier") if row.get("tier") is not None else row.get("curated_tier")
            games_raw = row.get("games") if row.get("games") is not None else row.get("play")
            games = int(games_raw) if games_raw is not None and str(games_raw).isdigit() else None

            # Tier-only row: preserve non-statistical metadata only
            if games is None and tier is not None:
                _append_unique(
                    Observation(
                        source_id=source_id,
                        overlap_group=overlap_group,
                        source_priority=source_priority,
                        is_disjoint_verified=False,
                        mode=mode,
                        metric_type=metric,  # type: ignore[arg-type]
                        patch=patch,
                        champion_id=champion_id,
                        entity_type=entity_type,  # type: ignore[arg-type]
                        entity_id=eid,
                        slot_key=slot_key,
                        curated_tier=str(tier),
                    )
                )
                continue

            if metric == "avg_placement":
                mean_p = row.get("mean_placement") or row.get("avg_placement") or row.get("avg_place")
                if games is None or mean_p is None:
                    rejected += 1
                    continue
                _append_unique(
                    Observation(
                        source_id=source_id,
                        overlap_group=overlap_group,
                        source_priority=source_priority,
                        is_disjoint_verified=False,
                        mode=mode,
                        metric_type="avg_placement",
                        patch=patch,
                        champion_id=champion_id,
                        entity_type=entity_type,  # type: ignore[arg-type]
                        entity_id=eid,
                        slot_key=slot_key,
                        games=games,
                        mean_placement=float(mean_p),
                        baseline_mean_placement=float(row.get("baseline_mean_placement", 4.5)),
                        curated_tier=str(tier) if tier is not None else None,
                    )
                )
                continue

            # Binary countable row
            wins = row.get("wins") if row.get("wins") is not None else row.get("win")
            if wins is None:
                rate_str = row.get("win_rate_str") or row.get("rate_str")
                if games is not None and rate_str is not None:
                    try:
                        wins = recover_unique_wins_from_rate(
                            rate_str=str(rate_str),
                            games=games,
                            rate_decimals=row.get("rate_decimals"),
                        )
                    except StatsValidationError:
                        rejected += 1
                        continue
                else:
                    rejected += 1
                    continue

            _append_unique(
                Observation(
                    source_id=source_id,
                    overlap_group=overlap_group,
                    source_priority=source_priority,
                    is_disjoint_verified=False,
                    mode=mode,
                    metric_type=metric,  # type: ignore[arg-type]
                    patch=patch,
                    champion_id=champion_id,
                    entity_type=entity_type,  # type: ignore[arg-type]
                    entity_id=eid,
                    slot_key=slot_key,
                    games=games,
                    wins=int(wins),
                    baseline_wins=int(baseline_wins) if baseline_wins is not None else None,
                    baseline_games=int(baseline_games) if baseline_games is not None else None,
                    baseline_rate=float(baseline_rate) if baseline_rate is not None else None,
                    champion_stratum_games=(
                        int(row["champion_stratum_games"])
                        if row.get("champion_stratum_games") is not None
                        else None
                    ),
                    pick_rate=float(row["pick_rate"]) if row.get("pick_rate") is not None else None,
                    curated_tier=str(tier) if tier is not None else None,
                )
            )

    _parse_group(list(page_props.get("augments", [])), "augment")
    _parse_group(list(page_props.get("items", [])), "item")

    # Mode landing pages (RSC) expose champion tier metadata without per-augment counts;
    # store as non-statistical tier metadata (never fabricated into wins/games).
    for crow in page_props.get("champion_tiers", []):
        ckey = canonicalize_slug(str(crow.get("key") or crow.get("name") or ""))
        tier_val = crow.get("tier")
        if not ckey:
            continue
        if tier_val is not None:
            _append_unique(
                Observation(
                    source_id=source_id,
                    overlap_group=overlap_group,
                    source_priority=source_priority,
                    is_disjoint_verified=False,
                    mode=mode,
                    metric_type=default_metric,  # type: ignore[arg-type]
                    patch=patch,
                    champion_id=ckey,
                    entity_type="augment",
                    entity_id="__champion_mode_tier__",
                    slot_key="all",
                    curated_tier=str(tier_val),
                )
            )
    return observations, rejected


class OpggStatsAdapter:
    """Live OP.GG SSR scraper implementing StatsAdapter v1 (`opgg_schema_v1`)."""

    schema_version = "v1"

    def __init__(
        self,
        *,
        adapter_id: str = "opgg",
        overlap_group: str = "riot_global_pool",
        source_priority: int = 30,
        is_disjoint_verified: bool = False,
        modes: list[GameMode] | None = None,
        champions: list[str] | None = None,
        delay_s: float = 0.5,
        timeout_s: float = 10.0,
        html_fixture: str | None = None,
    ) -> None:
        self.adapter_id = adapter_id
        self.overlap_group = overlap_group
        self.source_priority = source_priority
        self.is_disjoint_verified = is_disjoint_verified
        self.modes: list[GameMode] = modes or ["arena", "aram_mayhem"]
        self.champions = champions or []
        self.delay_s = max(0.0, delay_s)
        self.timeout_s = timeout_s
        self.html_fixture = html_fixture

    def parse_html(self, html: str, mode: GameMode) -> tuple[list[Observation], int]:
        props = extract_next_payloads_from_html(html)
        return parse_opgg_page_props(
            props,
            mode=mode,
            source_id=self.adapter_id,
            overlap_group=self.overlap_group,
            source_priority=self.source_priority,
        )

    def fetch(self) -> AdapterPayload:
        if self.html_fixture is not None:
            props = extract_next_payloads_from_html(self.html_fixture)
            mode: GameMode = props.get("mode", "aram_mayhem")
            obs, rej = parse_opgg_page_props(
                props,
                mode=mode,
                source_id=self.adapter_id,
                overlap_group=self.overlap_group,
                source_priority=self.source_priority,
            )
            return AdapterPayload(
                schema_version="v1",
                adapter_id=self.adapter_id,
                catalog_patch_hint=str(props.get("patch")) if props.get("patch") else None,
                observations=obs,
                adapter_metadata={
                    "opgg_schema": OPGG_SCHEMA_VERSION,
                    "rejected_rows": rej,
                },
            )

        all_obs: list[Observation] = []
        total_rejected = 0
        patch_hint: str | None = None
        headers = {"User-Agent": OPGG_USER_AGENT, "Accept": "text/html,application/xhtml+xml"}
        with httpx.Client(timeout=self.timeout_s, headers=headers, follow_redirects=True) as client:
            urls: list[tuple[GameMode, str]] = []
            for m in self.modes:
                base = OPGG_MODE_URLS[m]
                if self.champions:
                    for champ in self.champions:
                        urls.append((m, f"{base}/{champ}/build"))
                else:
                    urls.append((m, base))

            for idx, (m, url) in enumerate(urls):
                if idx > 0 and self.delay_s > 0.0:
                    time.sleep(self.delay_s)
                resp = client.get(url)
                resp.raise_for_status()
                props = extract_next_payloads_from_html(resp.text)
                if patch_hint is None and props.get("patch"):
                    patch_hint = str(props["patch"])
                obs, rej = parse_opgg_page_props(
                    props,
                    mode=m,
                    source_id=self.adapter_id,
                    overlap_group=self.overlap_group,
                    source_priority=self.source_priority,
                )
                all_obs.extend(obs)
                total_rejected += rej

        return AdapterPayload(
            schema_version="v1",
            adapter_id=self.adapter_id,
            catalog_patch_hint=patch_hint,
            observations=all_obs,
            adapter_metadata={
                "opgg_schema": OPGG_SCHEMA_VERSION,
                "rejected_rows": total_rejected,
            },
        )
