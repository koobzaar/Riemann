from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi.testclient import TestClient

from riemann.adapters.base import CommunityJsonAdapter, FixtureStatsAdapter
from riemann.adapters.cdragon import (
    build_patch_calendar_from_version,
    parse_cdragon_payloads,
)
from riemann.adapters.opgg import OpggStatsAdapter
from riemann.app import create_app
from riemann.config import Settings
from riemann.models import AdapterPayload, Observation, SyncRequest
from riemann.ocr import normalize_ocr_text, resolve_augment_text
from riemann.service import RiemannService

FIX_DIR = Path(__file__).resolve().parent / "fixtures"
GRAPH_PATH = Path(__file__).resolve().parents[1] / "data" / "relation_graph.json"


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        db_path=tmp_path / "cat.sqlite",
        graph_path=GRAPH_PATH,
        fixture_dir=FIX_DIR,
        block_outbound_on_inference=True,
        opgg_delay_s=0.0,
        opgg_timeout_s=5.0,
    )


def test_cdragon_community_json_and_opgg_parsers() -> None:
    bundle = parse_cdragon_payloads(
        default_payload=json.loads((FIX_DIR / "cdragon_raw_default.json").read_text(encoding="utf-8")),
        pt_br_payload=json.loads((FIX_DIR / "cdragon_raw_pt_br.json").read_text(encoding="utf-8")),
    )
    apex = next(a for a in bundle.augments if a.canonical_id == "apex_inventor")
    assert apex.names["en_us"] == "Apex Inventor"
    assert apex.names["pt_br"] == "Inventor Ápice"

    cj = CommunityJsonAdapter(FIX_DIR / "community_json_sample.json").fetch()
    assert cj.schema_version == "v1"
    assert len(cj.observations) == 2

    opgg = OpggStatsAdapter(html_fixture=(FIX_DIR / "opgg_raw_sample.html").read_text(encoding="utf-8")).fetch()
    assert opgg.adapter_metadata["rejected_rows"] == 1  # ambiguous 100000 @ 0.5000 rejected
    countable = [o for o in opgg.observations if o.games is not None]
    tier_only = [o for o in opgg.observations if o.games is None and o.curated_tier is not None]
    assert len(countable) == 2  # apex_inventor (unique 60/100) + heartsteel (112/200)
    assert len(tier_only) == 1  # weapon_of_destiny stored as tier-only metadata


def test_cdragon_empty_rejection_duplicate_merging_patch_calendar_and_opgg_rsc(tmp_path: Path) -> None:
    import pytest

    # 1. Corrupt empty catalog payload rejected (never overwrites valid snapshot)
    with pytest.raises(ValueError, match="corrupt_catalog_payload"):
        parse_cdragon_payloads(default_payload={}, pt_br_payload={})

    # 2. Real patch calendar parsing from CommunityDragon content-metadata version
    patch, cal = build_patch_calendar_from_version(
        "16.23.8406595+branch.releases-16-23.content.release"
    )
    assert patch == "16.23"
    assert cal == ["16.23", "16.22", "16.21", "16.20"]

    # 3. Live-shaped duplicate CommunityDragon augments (ARAM_ + Arena) & duplicate items merge cleanly
    dup_default = {
        "patch": "16.23",
        "patch_calendar": cal,
        "augments": [
            {"id": 10, "augmentNameId": "ApexInventor", "nameTRA": "Apex Inventor", "summaryDescription": "Grants 100 Item Haste."},
            {"id": 1210, "augmentNameId": "ARAM_ApexInventor", "nameTRA": "Apex Inventor", "summaryDescription": "Grants 100 Item Haste."},
        ],
        "items": [
            {"id": 3084, "name": "Heartsteel", "is_legendary": True},
            {"id": 33084, "name": "Heartsteel", "is_legendary": True},
        ],
        "champions": [
            {"id": 222, "name": "Jinx", "alias": "Jinx"},
            {"id": 222, "name": "Jinx", "alias": "Jinx"},
        ],
    }
    dup_pt = {
        "augments": [
            {"id": 10, "augmentNameId": "ApexInventor", "nameTRA": "Inventor Ápice"},
            {"id": 1210, "augmentNameId": "ARAM_ApexInventor", "nameTRA": "Inventor Ápice"},
        ],
        "items": [{"id": 3084, "name": "Coração de Aço"}],
        "champions": [{"id": 222, "name": "Jinx"}],
    }
    merged = parse_cdragon_payloads(default_payload=dup_default, pt_br_payload=dup_pt)
    assert len(merged.augments) == 1
    assert merged.augments[0].canonical_id == "apex_inventor"
    assert merged.augments[0].modes == ["aram_mayhem", "arena"]
    assert len(merged.items) == 1
    assert len(merged.champions) == 1

    # Sync merged bundle into SQLite cleanly (no UNIQUE constraint failure)
    s = _settings(tmp_path)
    svc = RiemannService(s, custom_catalog_loader=lambda _req: merged, custom_adapters={})
    resp_ok = svc._sync_blocking(SyncRequest(catalog_source="fixture"))
    assert resp_ok.catalog_patch == "16.23"

    # Subsequent empty catalog sync fails and preserves previous active snapshot
    svc.custom_catalog_loader = lambda _req: parse_cdragon_payloads(default_payload={}, pt_br_payload={})
    app = create_app(svc)
    client = TestClient(app)
    r_empty = client.post("/v1/catalog/sync", json={})
    assert r_empty.status_code == 502
    assert r_empty.json()["active_snapshot_preserved"] is True
    assert client.get("/health").json()["snapshot_id"] == resp_ok.snapshot_id

    # 4. OP.GG App Router RSC HTML extraction (including target-champion baseline isolation & mode propagation)
    rsc_inner = json.dumps(
        {
            "query": {"region": "global", "patch": "16.23"},
            "versions": ["16.23", "16.22"],
            "pathParams": {"mode": "arena", "slug": ["jinx", "build"]},
            "champions": [
                {"key": "aatrox", "name": "Aatrox", "tier": 4, "win_rate": 46.0, "pick_rate": 2.0},
                {"key": "jinx", "name": "Jinx", "tier": 1, "win_rate": 53.0, "pick_rate": 9.0},
            ],
            "augments": [
                {"name": "Apex Inventor", "play": "200", "win": 130, "win_rate": 65.0, "pick_rate": 12.0, "tier": "S"}
            ],
        }
    )
    escaped = json.dumps(rsc_inner)
    rsc_html = f"<html><body><script>self.__next_f.push([1,{escaped}])</script></body></html>"
    opgg_rsc = OpggStatsAdapter(html_fixture=rsc_html).fetch()
    assert opgg_rsc.catalog_patch_hint == "16.23"
    apex_obs = next(o for o in opgg_rsc.observations if o.entity_id == "apex_inventor")
    assert apex_obs.mode == "arena"
    assert apex_obs.metric_type == "top4"
    assert apex_obs.games == 200 and apex_obs.wins == 130
    assert abs((apex_obs.baseline_rate or 0.0) - 0.53) < 1e-6  # Jinx (0.53), NOT Aatrox (0.46) or augment (0.65)
    assert any(o.champion_id == "aatrox" and o.curated_tier == "4" for o in opgg_rsc.observations)

    # Case 2: augment win_rate appears before champion_summary -> must use champion_summary (0.50), not augment (0.65)
    rsc_inner2 = json.dumps(
        {
            "query": {"region": "global", "patch": "16.23"},
            "pathParams": {"mode": "aram_mayhem", "slug": ["jinx", "build"]},
            "augments": [{"name": "Apex Inventor", "play": 200, "win": 130, "win_rate": 65.0, "pick_rate": 12.0}],
            "champion_summary": {"win_rate": 50.0, "pick_rate": 8.0},
        }
    )
    rsc_html2 = f"<html><body><script>self.__next_f.push([1,{json.dumps(rsc_inner2)}])</script></body></html>"
    opgg_rsc2 = OpggStatsAdapter(html_fixture=rsc_html2).fetch()
    apex2 = next(o for o in opgg_rsc2.observations if o.entity_id == "apex_inventor")
    assert apex2.mode == "aram_mayhem"
    assert abs((apex2.baseline_rate or 0.0) - 0.50) < 1e-6


def test_double_sync_mid_adapter_failure_rollback_and_post_commit_restart_recovery(tmp_path: Path) -> None:
    class FailingMidwayAdapter:
        adapter_id = "bad_adapter"
        schema_version = "v1"
        overlap_group = "riot_global_pool"
        source_priority = 50
        is_disjoint_verified = False

        def fetch(self) -> AdapterPayload:
            # 10 valid rows followed by an invalid 11th row -> 0 rows committed from this adapter
            rows = [
                Observation(
                    source_id="bad_adapter",
                    mode="aram_mayhem",
                    metric_type="win",
                    patch="26.22",
                    champion_id="jinx",
                    entity_type="augment",
                    entity_id=f"neutral_aug",
                    cohort_id=f"c{i}",
                    games=100,
                    wins=55,
                    baseline_rate=0.50,
                )
                for i in range(10)
            ]
            rows.append(
                Observation(
                    source_id="bad_adapter",
                    mode="aram_mayhem",
                    metric_type="win",
                    patch="26.22",
                    champion_id="jinx",
                    entity_type="augment",
                    entity_id="neutral_aug",
                    cohort_id="c10",
                    games=100,
                    wins=150,  # INVALID!
                    baseline_rate=0.50,
                )
            )
            return AdapterPayload(adapter_id="bad_adapter", observations=rows)

    s = _settings(tmp_path)
    adapters = {
        "fixture": FixtureStatsAdapter(FIX_DIR / "stats_sample.json"),
        "bad_adapter": FailingMidwayAdapter(),
    }
    svc = RiemannService(s, custom_adapters=adapters)
    app = create_app(svc)
    client = TestClient(app)

    # Empty volume startup
    assert client.get("/health").json()["status"] == "degraded_empty_catalog"
    assert client.get("/v1/catalog/status").json()["snapshot_id"] is None
    assert client.post(
        "/v1/recommend/augments",
        json={"mode": "aram_mayhem", "champion_id": "jinx", "offered_augments": ["apex_inventor"]},
    ).status_code == 503

    # First sync: fixture succeeds, bad_adapter commits 0 rows
    r1 = client.post("/v1/catalog/sync", json={"adapters": ["fixture", "bad_adapter"]})
    assert r1.status_code == 200
    body1 = r1.json()
    by_name = {a["name"]: a for a in body1["adapters"]}
    assert by_name["fixture"]["status"] == "ok"
    assert by_name["bad_adapter"]["status"] == "error"
    assert by_name["bad_adapter"]["rows_loaded"] == 0
    snap1_id = body1["snapshot_id"]
    status1 = client.get("/v1/catalog/status").json()
    rec1 = client.post(
        "/v1/recommend/augments",
        json={"mode": "aram_mayhem", "champion_id": "dr_mundo", "slot_index": 1, "offered_augments": ["apex_inventor"]},
    ).json()

    # Second sync preserves exact counts, translations, and posterior values
    r2 = client.post("/v1/catalog/sync", json={"adapters": ["fixture"]})
    assert r2.status_code == 200
    status2 = client.get("/v1/catalog/status").json()
    assert status2["counts"] == status1["counts"]
    rec2 = client.post(
        "/v1/recommend/augments",
        json={"mode": "aram_mayhem", "champion_id": "dr_mundo", "slot_index": 1, "offered_augments": ["apex_inventor"]},
    ).json()
    assert rec2["recommendations"][0]["association_mean_logit"] == rec1["recommendations"][0]["association_mean_logit"]

    # Corrupt catalog sync rolls back and preserves previous active snapshot
    snap_before_corrupt = r2.json()["snapshot_id"]
    svc.custom_catalog_loader = lambda _req: (_ for _ in ()).throw(RuntimeError("corrupt catalog"))
    r_bad_cat = client.post("/v1/catalog/sync", json={"adapters": ["fixture"]})
    assert r_bad_cat.status_code == 502
    assert r_bad_cat.json()["active_snapshot_preserved"] is True
    assert client.get("/health").json()["snapshot_id"] == snap_before_corrupt

    # Post-commit cache swap failure -> returns 500, marks degraded_cache_reload_required,
    # keeps serving snapshot A in-memory, and restart loads durable snapshot B from SQLite!
    svc.custom_catalog_loader = None
    svc._fail_cache_swap_once = True
    r_swap_fail = client.post("/v1/catalog/sync", json={"adapters": ["fixture"]})
    assert r_swap_fail.status_code == 500
    assert r_swap_fail.json()["code"] == "cache_swap_failed"
    assert r_swap_fail.json()["active_snapshot_preserved"] is True
    assert client.get("/health").json()["status"] == "degraded_cache_reload_required"
    assert client.get("/health").json()["snapshot_id"] == snap_before_corrupt

    # Simulated process restart loads durable active_snapshot B committed to SQLite
    restarted_svc = RiemannService(s)
    assert restarted_svc.health().status == "ok"
    assert restarted_svc.health().snapshot_id != snap_before_corrupt
    assert restarted_svc.health().snapshot_id == restarted_svc.store.get_active_snapshot_id()


def test_concurrent_sync_returns_409(tmp_path: Path) -> None:
    s = _settings(tmp_path)
    svc = RiemannService(s)

    async def _run() -> int:
        await svc._sync_lock.acquire()
        try:
            app = create_app(svc)
            client = TestClient(app)
            resp = client.post("/v1/catalog/sync", json={})
            return resp.status_code
        finally:
            svc._sync_lock.release()

    assert asyncio.run(_run()) == 409


def test_ocr_normalization_thresholds_ambiguity_and_api_validation(tmp_path: Path) -> None:
    # Verify I -> l BEFORE lowercasing
    assert normalize_ocr_text("Apex Inventor") == normalize_ocr_text("Apex lnventor")
    assert normalize_ocr_text("Inventor Ápice") == normalize_ocr_text("lnventor Apice")

    svc = RiemannService(_settings(tmp_path))
    app = create_app(svc)
    client = TestClient(app)
    assert client.post("/v1/catalog/sync", json={}).status_code == 200

    # EN & PT-BR positive OCR resolutions
    r_en = client.post("/v1/resolve/augment", json={"text": "Apex lnventor", "locale": "en_us", "mode": "aram_mayhem"}).json()
    assert r_en["resolved"] is True and r_en["canonical_id"] == "apex_inventor" and r_en["confidence"] >= 0.80

    r_pt = client.post("/v1/resolve/augment", json={"text": "lnventor Ápice", "locale": "pt_br", "mode": "aram_mayhem"}).json()
    assert r_pt["resolved"] is True and r_pt["canonical_id"] == "apex_inventor" and r_pt["confidence"] >= 0.80

    r_wod = client.post("/v1/resolve/augment", json={"text": "Arma do Destino", "locale": "pt_br", "mode": "aram_mayhem"}).json()
    assert r_wod["resolved"] is True and r_wod["canonical_id"] == "weapon_of_destiny"

    # EN + PT-BR same-ID deduplication & en_us tie-breaker (no false ambiguity!)
    same_id_entries = [
        ("apex_inventor", "en_us", "Apex Inventor", normalize_ocr_text("Apex Inventor")),
        ("apex_inventor", "pt_br", "Apex Inventor", normalize_ocr_text("Apex Inventor")),
    ]
    res_same = resolve_augment_text(text="Apex Inventor", locale="auto", entries=same_id_entries)
    assert res_same.resolved is True and res_same.detected_locale == "en_us"

    # Single-candidate catalog resolves cleanly above 0.80
    res_single = resolve_augment_text(text="Apex Inventor", locale="en_us", entries=[same_id_entries[0]])
    assert res_single.resolved is True

    # Ambiguous match against two distinct IDs ("Critical S" vs Critical Strikes / Critical Shield)
    r_amb = client.post("/v1/resolve/augment", json={"text": "Critical S", "locale": "en_us", "mode": "aram_mayhem"}).json()
    assert r_amb["resolved"] is False and r_amb["reason"] == "ambiguous_match"
    assert len(r_amb["candidates"]) >= 2

    # Unrecognized noise
    r_noise = client.post("/v1/resolve/augment", json={"text": "zzzzzz", "locale": "auto", "mode": "aram_mayhem"}).json()
    assert r_noise["resolved"] is False and r_noise["reason"] == "below_confidence_threshold"

    # Mode isolation: arena-only augment rejected in aram_mayhem
    r_mode = client.post(
        "/v1/recommend/augments",
        json={"mode": "aram_mayhem", "champion_id": "jinx", "offered_augments": ["arena_only_aug"]},
    )
    assert r_mode.status_code == 422
    assert r_mode.json()["code"] == "augment_invalid_for_mode"

    # Partial OCR resolution -> 200 with unresolved_inputs
    r_part = client.post(
        "/v1/recommend/augments",
        json={"mode": "aram_mayhem", "champion_id": "jinx", "offered_augments": ["zzzzzz", "apex_inventor"]},
    )
    assert r_part.status_code == 200
    assert len(r_part.json()["recommendations"]) == 1
    assert r_part.json()["unresolved_inputs"][0]["raw_input"] == "zzzzzz"

    # Normalized 422 checks
    for bad_payload in [
        {"mode": "aram_mayhem", "champion_id": "jinx", "offered_augments": ["zzzzzz"]},
        {"mode": "aram_mayhem", "champion_id": "unknown_champ", "offered_augments": ["apex_inventor"]},
        {"mode": "aram_mayhem", "champion_id": "jinx", "chosen_augments": ["apex_inventor", "apex_inventor"], "offered_augments": ["neutral_aug"]},
        {"mode": "aram_mayhem", "champion_id": "jinx", "chosen_augments": ["apex_inventor"], "offered_augments": ["apex_inventor"]},
        {"mode": "aram_mayhem", "champion_id": "jinx", "observed_crit_chance": -0.1, "offered_augments": ["apex_inventor"]},
    ]:
        rr = client.post("/v1/recommend/augments", json=bad_payload)
        assert rr.status_code == 422
        assert set(rr.json().keys()) == {"code", "detail", "field", "active_snapshot_preserved"}
