from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from google import genai

from riemann.app import create_app
from riemann.config import Settings
from riemann.service import RiemannService


BUILD = {
    "mode": "aram_mayhem",
    "champion_id": "jinx",
    "chosen_augments": ["apex_inventor"],
    "locale": "pt_br",
}


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    repo = Path(__file__).resolve().parents[1]
    return Settings(
        db_path=tmp_path / "gemini.sqlite",
        graph_path=repo / "data" / "relation_graph.json",
        fixture_dir=repo / "tests" / "fixtures",
        block_outbound_on_inference=True,
        opgg_delay_s=0,
        opgg_timeout_s=5,
    )


class MockGemini:
    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.requests: list[dict] = []
        self.responses: list[str | int] = []
        real_client = genai.Client

        def create_client(*, http_options, **kwargs):
            transport = httpx.MockTransport(self.respond)
            options = http_options.model_copy(update={"httpx_client": httpx.Client(transport=transport)})
            return real_client(http_options=options, **kwargs)

        monkeypatch.setattr(genai, "Client", create_client)

    def respond(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(request.content))
        response = self.responses.pop(0)
        if isinstance(response, int):
            return httpx.Response(response, json={"error": {"message": "private provider details"}})
        return httpx.Response(
            200,
            json={"candidates": [{
                "content": {"role": "model", "parts": [{"text": response}]},
                "finishReason": "STOP",
            }]},
        )


def test_mechanics_enrichment_and_build_explanation(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    gemini = MockGemini(monkeypatch)
    settings = replace(settings, gemini_api_key="test-key", gemini_model="test-model")
    client = TestClient(create_app(RiemannService(settings)))
    assert client.post("/v1/catalog/sync").json()["mechanics_enrichment"] is None
    original_ranking = client.post("/v1/recommend/items", json=BUILD).json()
    assert gemini.requests == []

    note = {
        "augment_id": "apex_inventor",
        "source_quote": "Grants 100 Item Haste.",
        "note": "Item cooldowns benefit from the 100 Item Haste grant.",
    }
    gemini.responses.append(json.dumps({"notes": [note]}))
    synced = client.post("/v1/catalog/sync", json={"enrich_mechanics": True})
    assert synced.status_code == 200
    snapshot_id = synced.json()["snapshot_id"]
    enrichment = {"status": "ok", "model": "test-model", "notes": [note]}
    assert synced.json()["mechanics_enrichment"] == enrichment
    assert gemini.requests[0]["generationConfig"]["responseMimeType"] == "application/json"

    restarted = TestClient(create_app(RiemannService(settings)))
    assert restarted.get("/v1/catalog/status").json()["mechanics_enrichment"] == enrichment
    ranking = restarted.post("/v1/recommend/items", json=BUILD).json()
    assert ranking.pop("snapshot_id") == snapshot_id
    original_ranking.pop("snapshot_id")
    assert ranking == original_ranking
    assert len(gemini.requests) == 1

    gemini.responses.append("A Invenção Ápice reduz a recarga dos itens.")
    explanation = restarted.post("/v1/explain/build", json=BUILD)
    assert explanation.status_code == 200
    assert explanation.json() == {
        "snapshot_id": snapshot_id,
        "model": "test-model",
        "explanation": "A Invenção Ápice reduz a recarga dos itens.",
    }
    prompt = json.loads(gemini.requests[1]["contents"][0]["parts"][0]["text"])
    assert prompt["locale"] == "pt_br"
    assert prompt["mechanics_notes"] == [note]
    assert prompt["ranking"]["recommendations"] == ranking["recommendations"][:5]


def test_gemini_failure_leaves_inference_available(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    gemini = MockGemini(monkeypatch)
    client = TestClient(create_app(RiemannService(settings)))
    assert client.post("/v1/explain/build", json=BUILD).json()["code"] == "gemini_unavailable"
    sync = client.post("/v1/catalog/sync", json={"enrich_mechanics": True})
    assert sync.status_code == 200
    assert sync.json()["mechanics_enrichment"]["status"] == "unavailable"
    assert gemini.requests == []

    client = TestClient(create_app(RiemannService(replace(settings, gemini_api_key="test-key"))))
    fake_note = {"augment_id": "apex_inventor", "source_quote": "Imaginary quote", "note": "Unsupported"}
    gemini.responses.extend([json.dumps({"notes": [fake_note]}), 503])
    sync = client.post("/v1/catalog/sync", json={"enrich_mechanics": True})
    assert sync.status_code == 200
    assert sync.json()["mechanics_enrichment"]["status"] == "error"
    assert sync.json()["mechanics_enrichment"]["notes"] == []
    assert client.post("/v1/recommend/items", json=BUILD).status_code == 200
    response = client.post("/v1/explain/build", json=BUILD)
    assert response.status_code == 502
    assert response.json()["code"] == "gemini_explanation_failed"
    assert "private provider details" not in response.text
    assert len(gemini.requests) == 2
