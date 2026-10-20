import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PRESETS_DIR = ROOT / "presets"


def load_preset(preset_id: str) -> dict:
    return json.loads((PRESETS_DIR / f"{preset_id}.json").read_text())


@pytest.fixture(scope="session")
def engine():
    from core.engine_mlx import get_engine
    return get_engine()
