from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from riemann.models import AdapterPayload, Observation


class StatsAdapter(Protocol):
    adapter_id: str
    schema_version: str
    overlap_group: str
    source_priority: int
    is_disjoint_verified: bool

    def fetch(self) -> AdapterPayload: ...


class FixtureStatsAdapter:
    schema_version = "v1"

    def __init__(
        self,
        fixture_path: Path,
        *,
        adapter_id: str = "fixture",
        overlap_group: str = "riot_global_pool",
        source_priority: int = 10,
        is_disjoint_verified: bool = False,
    ) -> None:
        self.fixture_path = fixture_path
        self.adapter_id = adapter_id
        self.overlap_group = overlap_group
        self.source_priority = source_priority
        self.is_disjoint_verified = is_disjoint_verified

    def fetch(self) -> AdapterPayload:
        raw = json.loads(self.fixture_path.read_text(encoding="utf-8"))
        obs_list: list[Observation] = []
        for row in raw.get("observations", []):
            merged = {
                "source_id": raw.get("source_id", self.adapter_id),
                "overlap_group": raw.get("overlap_group", self.overlap_group),
                "source_priority": raw.get("source_priority", self.source_priority),
                "is_disjoint_verified": raw.get("is_disjoint_verified", self.is_disjoint_verified),
                **row,
            }
            obs_list.append(Observation.model_validate(merged))
        return AdapterPayload(
            schema_version="v1",
            adapter_id=self.adapter_id,
            catalog_patch_hint=raw.get("catalog_patch_hint"),
            observations=obs_list,
            adapter_metadata={"path": str(self.fixture_path)},
        )


class CommunityJsonAdapter:
    """Reads versioned v1 community stats JSON files."""

    schema_version = "v1"

    def __init__(
        self,
        json_path: Path,
        *,
        adapter_id: str = "community_json",
        overlap_group: str = "riot_global_pool",
        source_priority: int = 20,
        is_disjoint_verified: bool = False,
    ) -> None:
        self.json_path = json_path
        self.adapter_id = adapter_id
        self.overlap_group = overlap_group
        self.source_priority = source_priority
        self.is_disjoint_verified = is_disjoint_verified

    def fetch(self) -> AdapterPayload:
        raw = json.loads(self.json_path.read_text(encoding="utf-8"))
        if raw.get("schema_version") != "v1":
            raise ValueError(
                f"Unsupported community_json schema_version: {raw.get('schema_version')!r}"
            )
        source_id = str(raw.get("source_id", self.adapter_id))
        overlap_group = str(raw.get("overlap_group", self.overlap_group))
        priority = int(raw.get("source_priority", self.source_priority))
        disjoint = bool(raw.get("is_disjoint_verified", self.is_disjoint_verified))
        obs_list = [
            Observation.model_validate(
                {
                    "source_id": source_id,
                    "overlap_group": overlap_group,
                    "source_priority": priority,
                    "is_disjoint_verified": disjoint,
                    **row,
                }
            )
            for row in raw.get("observations", [])
        ]
        return AdapterPayload(
            schema_version="v1",
            adapter_id=self.adapter_id,
            catalog_patch_hint=raw.get("catalog_patch_hint"),
            observations=obs_list,
            adapter_metadata={"path": str(self.json_path)},
        )
