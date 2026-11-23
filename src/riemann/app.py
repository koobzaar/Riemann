from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from riemann.config import Settings
from riemann.models import (
    CatalogStatusResponse,
    HealthResponse,
    NormalizedError,
    RecommendAugmentsRequest,
    RecommendAugmentsResponse,
    RecommendItemsRequest,
    RecommendItemsResponse,
    ResolveAugmentRequest,
    ResolveAugmentResponse,
    SyncRequest,
    SyncResponse,
)
from riemann.network_guard import OutboundNetworkBlockedError
from riemann.recommender import DomainValidationError
from riemann.service import RiemannService, ServiceError


def create_app(service: RiemannService | None = None) -> FastAPI:
    svc = service if service is not None else RiemannService(Settings.from_env())
    app = FastAPI(
        title="Riemann Inference Backend",
        version="0.1.0",
        description="Statistical dynamic recommendation engine for League of Legends ARAM: Mayhem and Arena",
    )
    app.state.service = svc

    @app.exception_handler(ServiceError)
    async def handle_service_error(_req: Request, exc: ServiceError) -> JSONResponse:
        payload = NormalizedError(
            code=exc.code,
            detail=exc.detail,
            field=exc.field,
            active_snapshot_preserved=exc.active_snapshot_preserved,
        )
        return JSONResponse(status_code=exc.status_code, content=payload.model_dump())

    @app.exception_handler(DomainValidationError)
    async def handle_domain_validation(_req: Request, exc: DomainValidationError) -> JSONResponse:
        payload = NormalizedError(code=exc.code, detail=exc.detail, field=exc.field)
        return JSONResponse(status_code=422, content=payload.model_dump())

    @app.exception_handler(RequestValidationError)
    async def handle_request_validation(_req: Request, exc: RequestValidationError) -> JSONResponse:
        first = exc.errors()[0] if exc.errors() else {}
        loc = ".".join(str(p) for p in first.get("loc", []) if p != "body") or None
        msg = str(first.get("msg", "Validation error"))
        code = "contradictory_augment_binding" if "contradicts" in msg else "validation_error"
        payload = NormalizedError(code=code, detail=msg, field=loc)
        return JSONResponse(status_code=422, content=payload.model_dump())

    @app.exception_handler(OutboundNetworkBlockedError)
    async def handle_outbound_blocked(
        _req: Request, exc: OutboundNetworkBlockedError
    ) -> JSONResponse:
        payload = NormalizedError(code="outbound_network_blocked", detail=str(exc))
        return JSONResponse(status_code=500, content=payload.model_dump())

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_exc(_req: Request, exc: StarletteHTTPException) -> JSONResponse:
        payload = NormalizedError(code=f"http_{exc.status_code}", detail=str(exc.detail))
        return JSONResponse(status_code=exc.status_code, content=payload.model_dump())

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return svc.health()

    @app.get("/v1/catalog/status", response_model=CatalogStatusResponse)
    def catalog_status() -> CatalogStatusResponse:
        return svc.catalog_status()

    @app.post("/v1/catalog/sync", response_model=SyncResponse)
    async def catalog_sync(req: SyncRequest | None = None) -> SyncResponse:
        return await svc.sync(req or SyncRequest())

    @app.post("/v1/resolve/augment", response_model=ResolveAugmentResponse)
    def resolve_augment(req: ResolveAugmentRequest) -> ResolveAugmentResponse:
        return svc.resolve_augment(req)

    @app.post("/v1/recommend/augments", response_model=RecommendAugmentsResponse)
    def post_recommend_augments(req: RecommendAugmentsRequest) -> RecommendAugmentsResponse:
        return svc.recommend_augments(req)

    @app.post("/v1/recommend/items", response_model=RecommendItemsResponse)
    def post_recommend_items(req: RecommendItemsRequest) -> RecommendItemsResponse:
        return svc.recommend_items(req)

    return app


def get_app() -> FastAPI:
    return create_app()

