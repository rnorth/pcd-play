import pytest

from core.schema import StructuredSchema

CLAIM_TYPE = {
    "claim_type": {
        "type": "enum",
        "description": "Kind of loss being claimed",
        "choices": ["COLLISION", "THEFT", "FIRE", "WEATHER", "VANDALISM"],
    }
}


@pytest.mark.parametrize("context, expected", [
    ("Rear-end collision on the A40 on 14 March 2026, two passengers, no injuries.", "COLLISION"),
    ("Flood water from the storm got into the cabin; the car is written off.", "WEATHER"),
])
def test_enum_not_won_by_likelier_first_token(engine, context, expected):
    # Issue #3: unlisted, FIRE (F|IRE) beat these on its first token alone.
    from core.engine_mlx import run_parallel_generation
    result = run_parallel_generation(context, StructuredSchema(CLAIM_TYPE))
    assert result["parsed_json"]["claim_type"]["value"] == expected
