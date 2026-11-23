from __future__ import annotations

import json
from pathlib import Path

import tempfile
from riemann.app import create_app
from riemann.config import Settings
from riemann.service import RiemannService


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out_path = root / "openapi.json"
    with tempfile.TemporaryDirectory() as tmp:
        s = Settings(
            db_path=Path(tmp) / "openapi.sqlite",
            graph_path=root / "data" / "relation_graph.json",
            fixture_dir=root / "tests" / "fixtures",
            block_outbound_on_inference=True,
            opgg_delay_s=0.0,
            opgg_timeout_s=5.0,
        )
        app = create_app(RiemannService(s))
        out_path.write_text(json.dumps(app.openapi(), indent=2), encoding="utf-8")
    print(f"Exported OpenAPI schema to {out_path}")


if __name__ == "__main__":
    main()
