from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ModelHyperparams:
    pi_small: float = 0.90
    tau_small: float = 0.10
    tau_large: float = 0.50
    patch_decay: float = 0.70
    placement_prior_weight: float = 100.0
    placement_default_baseline: float = 4.5
    crit_overcap_k: float = 0.32
    weapon_destiny_rush_bonus: float = 0.16
    corroboration_divergent_threshold: float = 0.15
    ocr_confidence_threshold: float = 0.80
    ocr_margin_threshold: float = 0.08
    min_baseline_games: int = 50
    low_pick_rate_threshold: float = 0.02
    sensitivity_base_stratified: float = 0.05
    sensitivity_base_unstratified: float = 0.08
    sensitivity_base_sparse: float = 0.12
    quadrature_tol: float = 1e-6


@dataclass(frozen=True)
class Settings:
    db_path: Path
    graph_path: Path
    fixture_dir: Path
    block_outbound_on_inference: bool
    opgg_delay_s: float
    opgg_timeout_s: float
    hyperparams: ModelHyperparams = ModelHyperparams()
    gemini_api_key: str | None = field(default=None, repr=False)
    gemini_model: str = "gemini-3.8-flash"

    @classmethod
    def from_env(cls) -> Settings:
        root = Path(__file__).resolve().parents[2]
        return cls(
            db_path=Path(os.getenv("RIEMANN_DB_PATH", str(root / "riemann.sqlite"))),
            graph_path=Path(
                os.getenv("RIEMANN_GRAPH_PATH", str(root / "data" / "relation_graph.json"))
            ),
            fixture_dir=Path(
                os.getenv("RIEMANN_FIXTURE_DIR", str(root / "tests" / "fixtures"))
            ),
            block_outbound_on_inference=os.getenv(
                "RIEMANN_BLOCK_OUTBOUND_ON_INFERENCE", "0"
            ).strip()
            in {"1", "true", "TRUE", "yes"},
            opgg_delay_s=float(os.getenv("RIEMANN_OPGG_DELAY_S", "0.5")),
            opgg_timeout_s=float(os.getenv("RIEMANN_OPGG_TIMEOUT_S", "10.0")),
            gemini_api_key=os.getenv("GEMINI_API_KEY", "").strip() or None,
            gemini_model=os.getenv("RIEMANN_GEMINI_MODEL", "").strip() or cls.gemini_model,
        )
