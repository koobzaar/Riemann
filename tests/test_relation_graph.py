from __future__ import annotations

from pathlib import Path

import pytest

from riemann.mechanics import (
    ItemGraphMeta,
    RelationGraph,
    extract_fallback_augment_mechanics,
    load_relation_graph,
)
from riemann.models import CatalogAugment

GRAPH_PATH = Path(__file__).resolve().parents[1] / "data" / "relation_graph.json"


def test_production_relation_graph_minimum_coverage_and_integrity() -> None:
    graph = load_relation_graph(GRAPH_PATH)
    assert len(graph.augments) >= 40
    legendary_count = sum(1 for item in graph.items.values() if item.is_legendary)
    assert legendary_count >= 50
    assert len(graph.edges) >= 200

    all_tags = {t for aug in graph.augments.values() for t in aug.mechanic_tags}
    required_families = {
        "grants_crit_chance",
        "scales_with_crit",
        "grants_on_hit",
        "grants_item_haste",
        "empowers_first_legendary",
        "burn_dot",
        "heal_shield",
        "ability_haste",
        "hp_stacking",
    }
    assert required_families.issubset(all_tags)


def test_relation_graph_rejects_cycles_and_dangling_refs() -> None:
    with pytest.raises(ValueError, match="Cycle"):
        RelationGraph(
            graph_version="bad",
            items={
                "a": ItemGraphMeta(canonical_id="a", transformed_from="b"),
                "b": ItemGraphMeta(canonical_id="b", transformed_from="a"),
            },
        )

    with pytest.raises(ValueError, match="unknown"):
        RelationGraph(
            graph_version="bad",
            edges=[
                {
                    "edge_id": "e1",
                    "edge_type": "item_pair_synergy",
                    "source_entity_id": "nonexistent_item",
                    "target_tag": "hp_stacking",
                    "weight": 0.1,
                }
            ],
        )


def test_fallback_extractor_extracts_exact_tags_and_numeric_values() -> None:
    aug_haste = CatalogAugment(
        canonical_id="uncurated_haste_aug",
        modes=["aram_mayhem"],
        names={"en_us": "Super Inventor", "pt_br": "Super Inventor"},
        description="Grants 100 Item Haste and empowers your items.",
    )
    meta_h = extract_fallback_augment_mechanics(aug_haste)
    assert "grants_item_haste" in meta_h.mechanic_tags
    assert meta_h.item_haste == 100.0

    aug_crit = CatalogAugment(
        canonical_id="uncurated_crit_aug",
        modes=["arena"],
        names={"en_us": "Sharp Edge", "pt_br": "Fio Afiado"},
        description="Grants 25% Critical Strike Chance.",
    )
    meta_c = extract_fallback_augment_mechanics(aug_crit)
    assert "grants_crit_chance" in meta_c.mechanic_tags
    assert abs(meta_c.crit_chance - 0.25) < 1e-9

    aug_plain = CatalogAugment(
        canonical_id="uncurated_plain_aug",
        modes=["arena"],
        names={"en_us": "Plain Walk", "pt_br": "Caminhada"},
        description="Gain 15 movement speed out of combat.",
    )
    meta_p = extract_fallback_augment_mechanics(aug_plain)
    assert meta_p.mechanic_tags == []
    assert meta_p.item_haste == 0.0
    assert meta_p.crit_chance == 0.0
