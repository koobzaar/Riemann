from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

from riemann.adapters.base import CommunityJsonAdapter, FixtureStatsAdapter, StatsAdapter
from riemann.adapters.cdragon import fetch_live_cdragon_catalog, parse_cdragon_payloads
from riemann.adapters.opgg import OpggStatsAdapter
from riemann.config import Settings
from riemann.gemini import GeminiClient, GeminiError
from riemann.mechanics import (
    RelationGraph,
    compute_reference_prior_mean,
    effective_augment_meta,
    effective_champion_meta,
    effective_item_meta,
    load_relation_graph,
)
from riemann.models import (
    AdapterRunReport,
    BuildExplanationResponse,
    CatalogBundle,
    CatalogStatusResponse,
    GameMode,
    HealthResponse,
    MechanicsEnrichment,
    ModeCounts,
    Observation,
    PrecomputedPosterior,
    PublicLocale,
    RecommendAugmentsRequest,
    RecommendAugmentsResponse,
    RecommendItemsRequest,
    RecommendItemsResponse,
    ResolveAugmentRequest,
    ResolveAugmentResponse,
    SyncRequest,
    SyncResponse,
)
from riemann.network_guard import inference_no_outbound_context
from riemann.ocr import normalize_ocr_text, resolve_augment_text
from riemann.recommender import recommend_augments, recommend_items
from riemann.stats_engine import (
    ValidatedBinaryObs,
    ValidatedPlacementObs,
    compute_entity_posterior,
    validate_observations,
)
from riemann.storage import LoadedSnapshot, SnapshotStore


class ServiceError(RuntimeError):
    def __init__(
        self,
        status_code: int,
        code: str,
        detail: str,
        *,
        field: str | None = None,
        active_snapshot_preserved: bool | None = None,
    ) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.code = code
        self.detail = detail
        self.field = field
        self.active_snapshot_preserved = active_snapshot_preserved


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_translations(catalog: CatalogBundle) -> list[tuple[str, str, PublicLocale, str, str]]:
    rows: list[tuple[str, str, PublicLocale, str, str]] = []
    for etype, items in (
        ("augment", catalog.augments),
        ("item", catalog.items),
        ("champion", catalog.champions),
    ):
        for ent in items:
            for loc, dname in ent.names.items():
                rows.append((etype, ent.canonical_id, loc, dname, normalize_ocr_text(dname)))
    return rows


class RiemannService:
    def __init__(
        self,
        settings: Settings,
        *,
        graph: RelationGraph | None = None,
        custom_adapters: dict[str, StatsAdapter] | None = None,
        custom_catalog_loader: Callable[[SyncRequest], CatalogBundle] | None = None,
        fail_cache_swap_once: bool = False,
    ) -> None:
        self.settings = settings
        self.gemini = (
            GeminiClient(settings.gemini_api_key, settings.gemini_model)
            if settings.gemini_api_key else None
        )
        self.store = SnapshotStore(settings.db_path)
        self.graph: RelationGraph = (
            graph if graph is not None else load_relation_graph(settings.graph_path)
        )
        self.custom_adapters = custom_adapters
        self.custom_catalog_loader = custom_catalog_loader
        self._fail_cache_swap_once = fail_cache_swap_once
        self._sync_lock = asyncio.Lock()
        self._degraded_cache_reload_required = False
        self._active_snapshot: LoadedSnapshot | None = self.store.load_active_snapshot()

    def reload_from_db(self) -> None:
        self._active_snapshot = self.store.load_active_snapshot()
        self._degraded_cache_reload_required = False

    def health(self) -> HealthResponse:
        snap = self._active_snapshot
        if self._degraded_cache_reload_required:
            status = "degraded_cache_reload_required"
        elif snap is None:
            status = "degraded_empty_catalog"
        else:
            status = "ok"
        return HealthResponse(
            status=status,  # type: ignore[arg-type]
            snapshot_id=snap.snapshot_id if snap is not None else None,
            catalog_patch=snap.catalog_patch if snap is not None else None,
            modes=["aram_mayhem", "arena"],
        )

    def catalog_status(self) -> CatalogStatusResponse:
        snap = self._active_snapshot
        if snap is None:
            return CatalogStatusResponse(
                snapshot_id=None,
                catalog_patch=None,
                graph_version=None,
                counts={"aram_mayhem": ModeCounts(), "arena": ModeCounts()},
                locales=["en_us", "pt_br"],
                adapters=[],
            )
        counts: dict[GameMode, ModeCounts] = {}
        for mode in ("aram_mayhem", "arena"):
            counts[mode] = ModeCounts(
                augments=sum(1 for (m, _) in snap.augments if m == mode),
                items=sum(1 for (m, _) in snap.items if m == mode),
                champions=sum(1 for (m, _) in snap.champions if m == mode),
                stats_rows=snap.observations_count.get(mode, 0),
                posteriors=sum(1 for k in snap.posteriors if k[0] == mode),
            )
        return CatalogStatusResponse(
            snapshot_id=snap.snapshot_id,
            catalog_patch=snap.catalog_patch,
            graph_version=snap.graph_version,
            counts=counts,
            locales=["en_us", "pt_br"],
            adapters=list(snap.adapter_reports),
            mechanics_enrichment=snap.mechanics_enrichment,
        )

    def _load_catalog(self, req: SyncRequest) -> CatalogBundle:
        if self.custom_catalog_loader is not None:
            return self.custom_catalog_loader(req)
        if req.catalog_source == "cdragon":
            return fetch_live_cdragon_catalog()
        if req.fixture_profile == "scale":
            scale_path = self.settings.fixture_dir / "scale_fixture.json"
            raw = json.loads(scale_path.read_text(encoding="utf-8"))
            return CatalogBundle.model_validate(raw["catalog"])
        def_path = self.settings.fixture_dir / "cdragon_raw_default.json"
        pt_path = self.settings.fixture_dir / "cdragon_raw_pt_br.json"
        return parse_cdragon_payloads(
            default_payload=json.loads(def_path.read_text(encoding="utf-8")),
            pt_br_payload=json.loads(pt_path.read_text(encoding="utf-8")),
        )

    def _resolve_adapters(self, req: SyncRequest) -> list[StatsAdapter]:
        if self.custom_adapters is not None:
            if req.adapters is None:
                return list(self.custom_adapters.values())
            selected: list[StatsAdapter] = []
            for name in req.adapters:
                if name not in self.custom_adapters:
                    raise ServiceError(
                        422,
                        "unknown_adapter",
                        f"Unknown adapter: {name!r}",
                        field="adapters",
                        active_snapshot_preserved=True,
                    )
                selected.append(self.custom_adapters[name])
            return selected

        available: dict[str, StatsAdapter] = {}
        if req.fixture_profile == "scale":
            scale_path = self.settings.fixture_dir / "scale_fixture.json"
            available["fixture"] = FixtureStatsAdapter(scale_path, adapter_id="fixture")
        else:
            stats_path = self.settings.fixture_dir / "stats_sample.json"
            if stats_path.exists():
                available["fixture"] = FixtureStatsAdapter(stats_path, adapter_id="fixture")
            cj_path = self.settings.fixture_dir / "community_json_sample.json"
            if cj_path.exists():
                available["community_json"] = CommunityJsonAdapter(cj_path)
            opgg_html_path = self.settings.fixture_dir / "opgg_raw_sample.html"
            if opgg_html_path.exists() and req.catalog_source == "fixture":
                available["opgg"] = OpggStatsAdapter(
                    html_fixture=opgg_html_path.read_text(encoding="utf-8")
                )
            else:
                available["opgg"] = OpggStatsAdapter(
                    delay_s=self.settings.opgg_delay_s,
                    timeout_s=self.settings.opgg_timeout_s,
                )

        if req.adapters is None:
            return [available["fixture"]] if "fixture" in available else list(available.values())
        out: list[StatsAdapter] = []
        for name in req.adapters:
            if name not in available:
                raise ServiceError(
                    422,
                    "unknown_adapter",
                    f"Unknown adapter: {name!r}",
                    field="adapters",
                    active_snapshot_preserved=True,
                )
            out.append(available[name])
        return out

    async def sync(self, req: SyncRequest) -> SyncResponse:
        if self._sync_lock.locked():
            raise ServiceError(
                409,
                "sync_in_progress",
                "Another catalog sync is already in progress",
                active_snapshot_preserved=True,
            )

        async with self._sync_lock:
            return await asyncio.to_thread(self._sync_blocking, req)

    def _sync_blocking(self, req: SyncRequest) -> SyncResponse:
        now = _utc_now()
        try:
            catalog = self._load_catalog(req)
        except ServiceError:
            raise
        except Exception as exc:
            raise ServiceError(
                502,
                "catalog_sync_failed",
                f"Failed to load catalog: {exc}",
                active_snapshot_preserved=True,
            ) from exc

        adapters = self._resolve_adapters(req)
        all_valid_obs: list[Observation] = []
        all_binary: list[ValidatedBinaryObs] = []
        all_placement: list[ValidatedPlacementObs] = []
        reports: list[AdapterRunReport] = []

        # Per-adapter atomic staging: if an adapter fails, 0 rows from it are committed.
        for adapter in adapters:
            try:
                payload = adapter.fetch()
                bin_obs, place_obs = validate_observations(
                    payload.observations, patch_calendar=catalog.patch_calendar
                )
                all_valid_obs.extend(payload.observations)
                all_binary.extend(bin_obs)
                all_placement.extend(place_obs)
                reports.append(
                    AdapterRunReport(
                        name=adapter.adapter_id,
                        status="ok",
                        last_sync_utc=now,
                        rows_loaded=len(bin_obs) + len(place_obs),
                        rejected_rows=int(payload.adapter_metadata.get("rejected_rows", 0)),
                        error=None,
                    )
                )
            except Exception as exc:
                reports.append(
                    AdapterRunReport(
                        name=adapter.adapter_id,
                        status="error",
                        last_sync_utc=now,
                        rows_loaded=0,
                        rejected_rows=0,
                        error=str(exc),
                    )
                )

        # Build lookup maps from catalog + graph for precomputing posteriors
        aug_map = {(m, a.canonical_id): a for a in catalog.augments for m in a.modes}
        item_map = {(m, i.canonical_id): i for i in catalog.items for m in i.modes}
        champ_map = {(m, c.canonical_id): c for c in catalog.champions for m in c.modes}

        # Group observations by (mode, champion_id, entity_type, entity_id, slot_key)
        primary_bin_groups: dict[tuple[GameMode, str, str, str, str], list[ValidatedBinaryObs]] = {}
        top1_groups: dict[tuple[GameMode, str, str, str, str], list[ValidatedBinaryObs]] = {}
        place_groups: dict[tuple[GameMode, str, str, str, str], list[ValidatedPlacementObs]] = {}
        keys: set[tuple[GameMode, str, str, str, str]] = set()

        for b in all_binary:
            k = (b.obs.mode, b.obs.champion_id, b.obs.entity_type, b.obs.entity_id, b.obs.slot_key)
            keys.add(k)
            if b.obs.metric_type == "top1":
                top1_groups.setdefault(k, []).append(b)
            else:
                primary_bin_groups.setdefault(k, []).append(b)
        for p in all_placement:
            k = (p.obs.mode, p.obs.champion_id, p.obs.entity_type, p.obs.entity_id, p.obs.slot_key)
            keys.add(k)
            place_groups.setdefault(k, []).append(p)

        posteriors: list[PrecomputedPosterior] = []
        for mode, champ_id, etype, eid, slot_key in sorted(keys):
            champ_cat = champ_map.get((mode, champ_id))
            if champ_cat is None:
                continue
            champ_meta = effective_champion_meta(champ_cat, self.graph)

            if etype == "augment":
                ent_cat = aug_map.get((mode, eid))
                if ent_cat is None:
                    continue
                ent_meta = effective_augment_meta(ent_cat, self.graph)
                rework_list = [
                    champ_cat.rework_patch,
                    ent_cat.rework_patch,
                    catalog.mode_rework_patches.get(mode),
                ]
            else:
                ent_cat_i = item_map.get((mode, eid))
                if ent_cat_i is None:
                    continue
                ent_meta = effective_item_meta(ent_cat_i, self.graph)
                rework_list = [
                    champ_cat.rework_patch,
                    ent_cat_i.rework_patch,
                    catalog.mode_rework_patches.get(mode),
                ]

            prior_mean = compute_reference_prior_mean(
                mode=mode,
                champion=champ_meta,
                entity_type=etype,  # type: ignore[arg-type]
                entity_meta=ent_meta,
                graph=self.graph,
            )
            posteriors.append(
                compute_entity_posterior(
                    mode=mode,
                    champion_id=champ_id,
                    entity_type=etype,
                    entity_id=eid,
                    slot_key=slot_key,
                    primary_metric_rows=primary_bin_groups.get((mode, champ_id, etype, eid, slot_key), []),
                    top1_rows=top1_groups.get((mode, champ_id, etype, eid, slot_key), []),
                    placement_rows=place_groups.get((mode, champ_id, etype, eid, slot_key), []),
                    patch_calendar=catalog.patch_calendar,
                    rework_patches=rework_list,
                    prior_mean_logit=prior_mean,
                    hyperparams=self.settings.hyperparams,
                )
            )

        snapshot_id = f"snap_{uuid.uuid4().hex[:12]}"
        translations = _build_translations(catalog)
        enrichment = self._enrich_mechanics(catalog) if req.enrich_mechanics else None
        try:
            self.store.commit_snapshot(
                snapshot_id=snapshot_id,
                catalog=catalog,
                graph_version=self.graph.graph_version,
                hyperparams_json=json.dumps(asdict(self.settings.hyperparams)),
                created_utc=now,
                translations_rows=translations,
                observations=all_valid_obs,
                posteriors=posteriors,
                adapter_reports=reports,
                mechanics_enrichment=enrichment,
            )
        except Exception as exc:
            raise ServiceError(
                422,
                "snapshot_commit_failed",
                f"Failed to commit snapshot: {exc}",
                active_snapshot_preserved=True,
            ) from exc

        # Atomic in-memory swap (after durable SQLite commit)
        try:
            if self._fail_cache_swap_once:
                self._fail_cache_swap_once = False
                raise RuntimeError("Simulated post-commit in-memory cache swap failure")
            loaded = self.store.load_snapshot(snapshot_id)
            self._active_snapshot = loaded
            self._degraded_cache_reload_required = False
        except Exception as exc:
            self._degraded_cache_reload_required = True
            raise ServiceError(
                500,
                "cache_swap_failed",
                f"SQLite committed {snapshot_id} durably, but in-memory swap failed: {exc}",
                active_snapshot_preserved=True,
            ) from exc

        return SyncResponse(
            snapshot_id=snapshot_id,
            catalog_patch=catalog.catalog_patch,
            active_snapshot_preserved=False,
            adapters=reports,
            mechanics_enrichment=enrichment,
        )

    def _enrich_mechanics(self, catalog: CatalogBundle) -> MechanicsEnrichment:
        if self.gemini is None:
            return MechanicsEnrichment(status="unavailable")
        try:
            notes = self.gemini.enrich_mechanics(catalog)
        except GeminiError:
            return MechanicsEnrichment(status="error", model=self.gemini.model)
        return MechanicsEnrichment(status="ok", model=self.gemini.model, notes=notes)

    def _require_snapshot(self) -> LoadedSnapshot:
        snap = self._active_snapshot
        if snap is None:
            raise ServiceError(
                503,
                "snapshot_not_ready",
                "No active catalog snapshot loaded. Run POST /v1/catalog/sync first.",
            )
        return snap

    def resolve_augment(self, req: ResolveAugmentRequest) -> ResolveAugmentResponse:
        snap = self._require_snapshot()
        with inference_no_outbound_context(enabled=self.settings.block_outbound_on_inference):
            mode_aug_ids = {cid for (m, cid) in snap.augments if m == req.mode}
            entries = [
                (cid, loc, dname, nname)
                for (etype, cid, loc), (dname, nname) in snap.translations.items()
                if etype == "augment" and cid in mode_aug_ids
            ]
            return resolve_augment_text(
                text=req.text,
                locale=req.locale,
                entries=entries,
                confidence_threshold=self.settings.hyperparams.ocr_confidence_threshold,
                margin_threshold=self.settings.hyperparams.ocr_margin_threshold,
            )

    def recommend_augments(self, req: RecommendAugmentsRequest) -> RecommendAugmentsResponse:
        snap = self._require_snapshot()
        with inference_no_outbound_context(enabled=self.settings.block_outbound_on_inference):
            return recommend_augments(
                snapshot=snap,
                graph=self.graph,
                request=req,
                hyperparams=self.settings.hyperparams,
            )

    def recommend_items(self, req: RecommendItemsRequest) -> RecommendItemsResponse:
        return self._recommend_items(self._require_snapshot(), req)

    def _recommend_items(
        self, snap: LoadedSnapshot, req: RecommendItemsRequest
    ) -> RecommendItemsResponse:
        with inference_no_outbound_context(enabled=self.settings.block_outbound_on_inference):
            return recommend_items(
                snapshot=snap,
                graph=self.graph,
                request=req,
                hyperparams=self.settings.hyperparams,
            )

    def explain_build(self, req: RecommendItemsRequest) -> BuildExplanationResponse:
        if self.gemini is None:
            raise ServiceError(503, "gemini_unavailable", "Set GEMINI_API_KEY to enable explanations")

        snap = self._require_snapshot()
        ranking = self._recommend_items(snap, req)
        ranking = ranking.model_copy(update={"recommendations": ranking.recommendations[:5]})
        locale: PublicLocale = "en_us" if req.locale == "auto" else req.locale
        champion = snap.champions[(req.mode, req.champion_id)]
        notes = snap.mechanics_enrichment.notes if snap.mechanics_enrichment else []
        data = {
            "catalog_patch": snap.catalog_patch,
            "locale": locale,
            "champion": champion.names.get(locale, champion.names["en_us"]),
            "chosen_augments": [
                snap.augments[(req.mode, augment_id)].names for augment_id in req.chosen_augments
            ],
            "current_items": [snap.items[(req.mode, item_id)].names for item_id in req.current_items],
            "mechanics_notes": [
                note.model_dump() for note in notes if note.augment_id in req.chosen_augments
            ],
            "ranking": ranking.model_dump(mode="json"),
        }
        try:
            explanation = self.gemini.explain_build(data)
        except GeminiError as exc:
            raise ServiceError(502, "gemini_explanation_failed", "Gemini could not explain the build") from exc
        return BuildExplanationResponse(
            snapshot_id=snap.snapshot_id, model=self.gemini.model, explanation=explanation
        )
