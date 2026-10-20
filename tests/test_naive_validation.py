from core.naive_validation import validate_naive_output
from core.schema import StructuredSchema

SCHEMA = StructuredSchema({
    "tier": {"type": "enum", "choices": ["LOW", "HIGH"]},
    "urgent": {"type": "boolean"},
    "when": {"type": "date"},
})


def test_valid_output_with_surrounding_text():
    r = validate_naive_output('Sure! {"tier": "LOW", "urgent": true, "when": null} hope that helps', SCHEMA)
    assert r["is_valid_json"] and r["schema_match"]
    assert r["parsed_json"] == {"tier": "LOW", "urgent": True, "when": None}
    assert r["missing_keys"] == [] and r["invalid_enums"] == [] and r["parse_error"] is None


def test_missing_key_and_invalid_enum():
    r = validate_naive_output('{"tier": "MEDIUM", "when": "2026-01-01"}', SCHEMA)
    assert r["is_valid_json"] and not r["schema_match"]
    assert r["missing_keys"] == ["urgent"]
    assert r["invalid_enums"] == ["tier=MEDIUM"]


def test_extra_keys_are_allowed():
    r = validate_naive_output('{"tier": "HIGH", "urgent": false, "when": null, "note": "x"}', SCHEMA)
    assert r["schema_match"]


def test_unparseable_output():
    r = validate_naive_output('{"tier": "LOW",', SCHEMA)
    assert not r["is_valid_json"] and not r["schema_match"]
    assert r["parsed_json"] is None and r["parse_error"]
