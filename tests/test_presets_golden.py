import json
import os
from pathlib import Path

import pytest

from conftest import PRESETS_DIR, load_preset
from core.schema import StructuredSchema

GOLDEN_DIR = Path(__file__).parent / "golden"
UPDATE = os.environ.get("UPDATE_GOLDEN") == "1"

PRESET_IDS = sorted(
    p.stem for p in PRESETS_DIR.glob("*.json")
    if UPDATE or (GOLDEN_DIR / p.name).exists()
)


@pytest.mark.parametrize("preset_id", PRESET_IDS)
def test_presets_match_golden(engine, preset_id):
    from core.engine_mlx import run_parallel_generation

    preset = load_preset(preset_id)
    result = run_parallel_generation(preset["context"], StructuredSchema(preset["schema"]))
    actual = {k: {"value": v["value"], "prob": v["prob"]} for k, v in result["parsed_json"].items()}
    golden_path = GOLDEN_DIR / f"{preset_id}.json"

    if UPDATE:
        golden_path.write_text(json.dumps(actual, indent=2) + "\n")
        pytest.skip("golden updated")

    golden = json.loads(golden_path.read_text())
    assert list(actual) == list(golden)
    for field, g in golden.items():
        assert actual[field]["value"] == g["value"], field
        assert abs(actual[field]["prob"] - g["prob"]) < 1e-3, field
