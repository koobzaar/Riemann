from __future__ import annotations

import math
import os
import subprocess
import time
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from riemann.app import create_app
from riemann.config import Settings
from riemann.models import (
    RecommendAugmentsResponse,
    RecommendItemsResponse,
    ResolveAugmentResponse,
)
from riemann.service import RiemannService

FIX_DIR = Path(__file__).resolve().parent / "fixtures"
GRAPH_PATH = Path(__file__).resolve().parents[1] / "data" / "relation_graph.json"


def _nearest_rank_p95(samples_ms: list[float]) -> float:
    ordered = sorted(samples_ms)
    idx = math.ceil(0.95 * len(ordered)) - 1  # 48th of 50 -> 0-based index 47
    return ordered[idx]


def _run_e2e_checks(client, *, compose_project: str | None = None) -> None:
    # 1. Empty volume startup
    h0 = client.get("/health")
    assert h0.status_code == 200
    assert h0.json()["status"] == "degraded_empty_catalog"

    st0 = client.get("/v1/catalog/status")
    assert st0.status_code == 200
    assert st0.json()["snapshot_id"] is None

    pre = client.post(
        "/v1/recommend/augments",
        json={"mode": "aram_mayhem", "champion_id": "jinx", "offered_augments": ["apex_inventor"]},
    )
    assert pre.status_code == 503
    assert pre.json()["code"] == "snapshot_not_ready"

    # 2. Sync realistic scale fixture (150 augs, 100 items, 160 champs, >=500 posteriors, >=200 graph edges)
    sync_resp = client.post(
        "/v1/catalog/sync",
        json={"catalog_source": "fixture", "fixture_profile": "scale", "adapters": ["fixture"]},
    )
    assert sync_resp.status_code == 200
    snap_id = sync_resp.json()["snapshot_id"]

    h1 = client.get("/health")
    assert h1.status_code == 200 and h1.json()["status"] == "ok" and h1.json()["snapshot_id"] == snap_id

    st1 = client.get("/v1/catalog/status").json()
    assert st1["counts"]["aram_mayhem"]["augments"] >= 148
    assert st1["counts"]["aram_mayhem"]["items"] >= 99
    assert st1["counts"]["aram_mayhem"]["champions"] >= 160
    assert st1["counts"]["aram_mayhem"]["posteriors"] >= 500

    # 3. Real container restart persistence (when running against Docker)
    if compose_project is not None:
        subprocess.run(
            ["docker", "compose", "-p", compose_project, "restart"],
            check=True,
            capture_output=True,
            text=True,
        )
        # Wait until healthy again
        for _ in range(30):
            try:
                hr = client.get("/health")
                if hr.status_code == 200 and hr.json()["status"] == "ok":
                    break
            except Exception:
                pass
            time.sleep(0.5)
        hr = client.get("/health")
        assert hr.status_code == 200
        assert hr.json()["snapshot_id"] == snap_id

    # 4. Functional + schema + cold/error path checks
    res_bad = client.post(
        "/v1/resolve/augment",
        json={"text": "zzzzzz", "locale": "auto", "mode": "aram_mayhem"},
    )
    assert res_bad.status_code == 200 and res_bad.json()["resolved"] is False

    req_resolve = {"text": "Apex lnventor", "locale": "en_us", "mode": "aram_mayhem"}
    req_augs = {
        "mode": "aram_mayhem",
        "champion_id": "jinx",
        "slot_index": 2,
        "chosen_augments": ["it_is_critical", "neutral_aug"],
        "current_items": ["infinity_edge", "phantom_dancer", "warmogs_armor"],
        "offered_augments": ["jeweled_gauntlet", "ethereal_weapon", "apex_inventor"],
        "top_item_pivots_k": 3,
    }
    req_items = {
        "mode": "aram_mayhem",
        "champion_id": "dr_mundo",
        "slot_index": 1,
        "chosen_augments": ["apex_inventor", "neutral_aug"],
        "current_items": ["warmogs_armor", "thornmail", "giants_belt"],
        "top_k": 10,
    }

    # 5. Latency benchmark: 10 warm-up + 50 sequential requests per endpoint; nearest-rank p95 < 50 ms
    endpoints = [
        ("/v1/resolve/augment", req_resolve, ResolveAugmentResponse),
        ("/v1/recommend/augments", req_augs, RecommendAugmentsResponse),
        ("/v1/recommend/items", req_items, RecommendItemsResponse),
    ]
    for path, payload, schema_cls in endpoints:
        for _ in range(10):
            r = client.post(path, json=payload)
            assert r.status_code == 200
            schema_cls.model_validate(r.json())

        timings_ms: list[float] = []
        for _ in range(50):
            t0 = time.perf_counter()
            r = client.post(path, json=payload)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            assert r.status_code == 200
            schema_cls.model_validate(r.json())
            timings_ms.append(dt_ms)

        p95 = _nearest_rank_p95(timings_ms)
        assert p95 < 50.0, f"{path} p95 latency {p95:.2f} ms exceeded 50 ms"


def test_api_e2e_and_scale_latency(tmp_path: Path) -> None:
    base_url = os.getenv("RIEMANN_E2E_BASE_URL")
    compose_project = os.getenv("RIEMANN_E2E_COMPOSE_PROJECT")

    if base_url:
        with httpx.Client(base_url=base_url, timeout=15.0) as client:
            _run_e2e_checks(client, compose_project=compose_project)
    else:
        settings = Settings(
            db_path=tmp_path / "e2e.sqlite",
            graph_path=GRAPH_PATH,
            fixture_dir=FIX_DIR,
            block_outbound_on_inference=True,
            opgg_delay_s=0.0,
            opgg_timeout_s=5.0,
        )
        svc = RiemannService(settings)
        app = create_app(svc)
        with TestClient(app) as client:
            _run_e2e_checks(client, compose_project=None)
