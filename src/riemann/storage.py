from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from riemann.models import (
    AdapterRunReport,
    CatalogAugment,
    CatalogBundle,
    CatalogChampion,
    CatalogItem,
    GameMode,
    MechanicsEnrichment,
    Observation,
    PrecomputedPosterior,
    PublicLocale,
)


SCHEMA_SQL = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS snapshots (
    snapshot_id TEXT PRIMARY KEY,
    catalog_patch TEXT NOT NULL,
    graph_version TEXT NOT NULL,
    patch_calendar_json TEXT NOT NULL,
    mode_rework_patches_json TEXT NOT NULL,
    hyperparams_json TEXT NOT NULL,
    created_utc TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS active_snapshot (
    singleton_id INTEGER PRIMARY KEY CHECK (singleton_id = 1),
    snapshot_id TEXT NOT NULL REFERENCES snapshots(snapshot_id)
);

CREATE TABLE IF NOT EXISTS entities (
    snapshot_id TEXT NOT NULL REFERENCES snapshots(snapshot_id) ON DELETE CASCADE,
    entity_type TEXT NOT NULL,
    canonical_id TEXT NOT NULL,
    mode TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    PRIMARY KEY (snapshot_id, entity_type, canonical_id, mode)
);

CREATE TABLE IF NOT EXISTS entity_translations (
    snapshot_id TEXT NOT NULL REFERENCES snapshots(snapshot_id) ON DELETE CASCADE,
    entity_type TEXT NOT NULL,
    canonical_id TEXT NOT NULL,
    locale TEXT NOT NULL,
    display_name TEXT NOT NULL,
    normalized_name TEXT NOT NULL,
    PRIMARY KEY (snapshot_id, entity_type, canonical_id, locale)
);

CREATE TABLE IF NOT EXISTS stats_observations (
    snapshot_id TEXT NOT NULL REFERENCES snapshots(snapshot_id) ON DELETE CASCADE,
    source_id TEXT NOT NULL,
    cohort_id TEXT NOT NULL,
    mode TEXT NOT NULL,
    metric_type TEXT NOT NULL,
    patch TEXT NOT NULL,
    champion_id TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    slot_key TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    PRIMARY KEY (
        snapshot_id, source_id, cohort_id, mode, metric_type,
        patch, champion_id, entity_type, entity_id, slot_key
    )
);

CREATE TABLE IF NOT EXISTS precomputed_posteriors (
    snapshot_id TEXT NOT NULL REFERENCES snapshots(snapshot_id) ON DELETE CASCADE,
    mode TEXT NOT NULL,
    champion_id TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    slot_key TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    PRIMARY KEY (snapshot_id, mode, champion_id, entity_type, entity_id, slot_key)
);

CREATE TABLE IF NOT EXISTS adapter_runs (
    snapshot_id TEXT NOT NULL REFERENCES snapshots(snapshot_id) ON DELETE CASCADE,
    adapter_name TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    PRIMARY KEY (snapshot_id, adapter_name)
);

CREATE TABLE IF NOT EXISTS mechanics_enrichments (
    snapshot_id TEXT PRIMARY KEY REFERENCES snapshots(snapshot_id) ON DELETE CASCADE,
    payload_json TEXT NOT NULL
);
"""


@dataclass(frozen=True)
class LoadedSnapshot:
    snapshot_id: str
    catalog_patch: str
    graph_version: str
    patch_calendar: tuple[str, ...]
    mode_rework_patches: dict[GameMode, str | None]
    augments: dict[tuple[GameMode, str], CatalogAugment]
    items: dict[tuple[GameMode, str], CatalogItem]
    champions: dict[tuple[GameMode, str], CatalogChampion]
    translations: dict[tuple[str, str, PublicLocale], tuple[str, str]]
    observations_count: dict[GameMode, int]
    posteriors: dict[tuple[GameMode, str, str, str, str], PrecomputedPosterior]
    adapter_reports: tuple[AdapterRunReport, ...]
    mechanics_enrichment: MechanicsEnrichment | None = None


class SnapshotStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = self._connect()
        try:
            conn.executescript(SCHEMA_SQL)
            conn.commit()
        finally:
            conn.close()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def get_active_snapshot_id(self) -> str | None:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT snapshot_id FROM active_snapshot WHERE singleton_id = 1"
            ).fetchone()
            return str(row["snapshot_id"]) if row is not None else None
        finally:
            conn.close()

    def commit_snapshot(
        self,
        *,
        snapshot_id: str,
        catalog: CatalogBundle,
        graph_version: str,
        hyperparams_json: str,
        created_utc: str,
        translations_rows: list[tuple[str, str, PublicLocale, str, str]],
        observations: list[Observation],
        posteriors: list[PrecomputedPosterior],
        adapter_reports: list[AdapterRunReport],
        mechanics_enrichment: MechanicsEnrichment | None = None,
    ) -> None:
        """Atomically persist all snapshot tables and update durable active_snapshot pointer."""
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE;")
            conn.execute(
                """
                INSERT INTO snapshots (
                    snapshot_id, catalog_patch, graph_version,
                    patch_calendar_json, mode_rework_patches_json,
                    hyperparams_json, created_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    snapshot_id,
                    catalog.catalog_patch,
                    graph_version,
                    json.dumps(catalog.patch_calendar),
                    json.dumps(catalog.mode_rework_patches),
                    hyperparams_json,
                    created_utc,
                ),
            )

            entity_rows: list[tuple[str, str, str, str, str]] = []
            for aug in catalog.augments:
                pj = aug.model_dump_json()
                for mode in aug.modes:
                    entity_rows.append((snapshot_id, "augment", aug.canonical_id, mode, pj))
            for item in catalog.items:
                pj = item.model_dump_json()
                for mode in item.modes:
                    entity_rows.append((snapshot_id, "item", item.canonical_id, mode, pj))
            for champ in catalog.champions:
                pj = champ.model_dump_json()
                for mode in champ.modes:
                    entity_rows.append((snapshot_id, "champion", champ.canonical_id, mode, pj))
            conn.executemany(
                """
                INSERT INTO entities (snapshot_id, entity_type, canonical_id, mode, payload_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                entity_rows,
            )

            conn.executemany(
                """
                INSERT INTO entity_translations (
                    snapshot_id, entity_type, canonical_id, locale, display_name, normalized_name
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (snapshot_id, etype, cid, loc, dname, nname)
                    for etype, cid, loc, dname, nname in translations_rows
                ],
            )

            conn.executemany(
                """
                INSERT INTO stats_observations (
                    snapshot_id, source_id, cohort_id, mode, metric_type,
                    patch, champion_id, entity_type, entity_id, slot_key, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        snapshot_id,
                        obs.source_id,
                        obs.cohort_id,
                        obs.mode,
                        obs.metric_type,
                        obs.patch,
                        obs.champion_id,
                        obs.entity_type,
                        obs.entity_id,
                        obs.slot_key,
                        obs.model_dump_json(),
                    )
                    for obs in observations
                ],
            )

            conn.executemany(
                """
                INSERT INTO precomputed_posteriors (
                    snapshot_id, mode, champion_id, entity_type, entity_id, slot_key, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        snapshot_id,
                        p.mode,
                        p.champion_id,
                        p.entity_type,
                        p.entity_id,
                        p.slot_key,
                        p.model_dump_json(),
                    )
                    for p in posteriors
                ],
            )

            conn.executemany(
                """
                INSERT INTO adapter_runs (snapshot_id, adapter_name, payload_json)
                VALUES (?, ?, ?)
                """,
                [(snapshot_id, r.name, r.model_dump_json()) for r in adapter_reports],
            )

            if mechanics_enrichment is not None:
                conn.execute(
                    "INSERT INTO mechanics_enrichments (snapshot_id, payload_json) VALUES (?, ?)",
                    (snapshot_id, mechanics_enrichment.model_dump_json()),
                )

            conn.execute(
                """
                INSERT INTO active_snapshot (singleton_id, snapshot_id)
                VALUES (1, ?)
                ON CONFLICT(singleton_id) DO UPDATE SET snapshot_id = excluded.snapshot_id
                """,
                (snapshot_id,),
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def load_active_snapshot(self) -> LoadedSnapshot | None:
        sid = self.get_active_snapshot_id()
        if sid is None:
            return None
        return self.load_snapshot(sid)

    def load_snapshot(self, snapshot_id: str) -> LoadedSnapshot:
        conn = self._connect()
        try:
            srow = conn.execute(
                "SELECT * FROM snapshots WHERE snapshot_id = ?", (snapshot_id,)
            ).fetchone()
            if srow is None:
                raise KeyError(f"Unknown snapshot_id: {snapshot_id}")

            augments: dict[tuple[GameMode, str], CatalogAugment] = {}
            items: dict[tuple[GameMode, str], CatalogItem] = {}
            champions: dict[tuple[GameMode, str], CatalogChampion] = {}

            for row in conn.execute(
                "SELECT entity_type, canonical_id, mode, payload_json FROM entities WHERE snapshot_id = ?",
                (snapshot_id,),
            ):
                mode: GameMode = row["mode"]
                cid = str(row["canonical_id"])
                etype = str(row["entity_type"])
                pj = str(row["payload_json"])
                if etype == "augment":
                    augments[(mode, cid)] = CatalogAugment.model_validate_json(pj)
                elif etype == "item":
                    items[(mode, cid)] = CatalogItem.model_validate_json(pj)
                elif etype == "champion":
                    champions[(mode, cid)] = CatalogChampion.model_validate_json(pj)

            translations: dict[tuple[str, str, PublicLocale], tuple[str, str]] = {}
            for row in conn.execute(
                """
                SELECT entity_type, canonical_id, locale, display_name, normalized_name
                FROM entity_translations WHERE snapshot_id = ?
                """,
                (snapshot_id,),
            ):
                translations[(str(row["entity_type"]), str(row["canonical_id"]), row["locale"])] = (
                    str(row["display_name"]),
                    str(row["normalized_name"]),
                )

            obs_counts: dict[GameMode, int] = {"aram_mayhem": 0, "arena": 0}
            for row in conn.execute(
                "SELECT mode, COUNT(*) AS cnt FROM stats_observations WHERE snapshot_id = ? GROUP BY mode",
                (snapshot_id,),
            ):
                obs_counts[row["mode"]] = int(row["cnt"])

            posteriors: dict[tuple[GameMode, str, str, str, str], PrecomputedPosterior] = {}
            for row in conn.execute(
                """
                SELECT mode, champion_id, entity_type, entity_id, slot_key, payload_json
                FROM precomputed_posteriors WHERE snapshot_id = ?
                """,
                (snapshot_id,),
            ):
                p = PrecomputedPosterior.model_validate_json(str(row["payload_json"]))
                posteriors[(p.mode, p.champion_id, p.entity_type, p.entity_id, p.slot_key)] = p

            adapter_reports: list[AdapterRunReport] = []
            for row in conn.execute(
                "SELECT payload_json FROM adapter_runs WHERE snapshot_id = ? ORDER BY adapter_name",
                (snapshot_id,),
            ):
                adapter_reports.append(AdapterRunReport.model_validate_json(str(row["payload_json"])))

            enrichment_row = conn.execute(
                "SELECT payload_json FROM mechanics_enrichments WHERE snapshot_id = ?",
                (snapshot_id,),
            ).fetchone()
            enrichment = (
                MechanicsEnrichment.model_validate_json(enrichment_row["payload_json"])
                if enrichment_row is not None else None
            )

            return LoadedSnapshot(
                snapshot_id=snapshot_id,
                catalog_patch=str(srow["catalog_patch"]),
                graph_version=str(srow["graph_version"]),
                patch_calendar=tuple(json.loads(str(srow["patch_calendar_json"]))),
                mode_rework_patches=json.loads(str(srow["mode_rework_patches_json"])),
                augments=augments,
                items=items,
                champions=champions,
                translations=translations,
                observations_count=obs_counts,
                posteriors=posteriors,
                adapter_reports=tuple(adapter_reports),
                mechanics_enrichment=enrichment,
            )
        finally:
            conn.close()
