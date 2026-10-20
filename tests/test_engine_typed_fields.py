from conftest import load_preset
from core.schema import StructuredSchema

TYPED = ("incident_date", "submitted_at", "repair_estimate_gbp", "passenger_count")


def run(context, schema_dict):
    from core.engine_mlx import run_parallel_generation
    return run_parallel_generation(context, StructuredSchema(schema_dict))


def values(result):
    return {k: v["value"] for k, v in result["parsed_json"].items()}


def test_insurance_claim_values(engine):
    preset = load_preset("insurance_claim")
    result = run(preset["context"], preset["schema"])
    assert values(result) == {
        "incident_date": "2026-03-14",
        "submitted_at": "2026-03-16T09:12:30",
        "repair_estimate_gbp": 2350.5,
        "passenger_count": 2,
        "claim_type": "collision",
        "injuries_reported": False,
    }
    for f in TYPED:
        assert {"type", "confidence", "p_null"} <= set(result["field_telemetry"][f])


def test_unrelated_text_is_null(engine):
    preset = load_preset("insurance_claim")
    text = "Hi team, the offsite is moving to next quarter, details to follow. Cheers, Sam"
    result = run(text, preset["schema"])
    for f in TYPED:
        assert result["parsed_json"][f]["value"] is None, f
        assert result["field_telemetry"][f]["p_null"] > 0.5, f


def test_single_generative_field(engine):
    preset = load_preset("insurance_claim")
    result = run(preset["context"], {"incident_date": preset["schema"]["incident_date"]})
    assert values(result) == {"incident_date": "2026-03-14"}


def test_collision_enum_alongside_number(engine, monkeypatch):
    # The parallel catalogue omits types, so a boolean stand-in plus a forced null rule
    # gives a byte-identical prompt through the pre-lockstep path: the enum/boolean
    # results (incl. the collision-resolved command_area) must not change.
    preset = load_preset("smart_home_voice")

    def schema_with(extra_type):
        schema = dict(preset["schema"])
        schema["brightness_pct"] = {"type": extra_type, "description": "Requested brightness percentage"}
        return schema

    typed = run(preset["context"], schema_with("number"))["parsed_json"]
    monkeypatch.setattr(StructuredSchema, "has_generative", property(lambda self: True))
    reference = run(preset["context"], schema_with("boolean"))["parsed_json"]

    for f in preset["schema"]:
        assert typed[f]["value"] == reference[f]["value"], f
        assert abs(typed[f]["prob"] - reference[f]["prob"]) < 1e-3, f
    assert typed["brightness_pct"]["value"] is None or isinstance(typed["brightness_pct"]["value"], int)


def test_naive_does_not_flag_typed_fields(engine):
    from core.engine_mlx import run_naive_generation

    preset = load_preset("insurance_claim")
    result = run_naive_generation(preset["context"], StructuredSchema(preset["schema"]))
    flagged = {entry.split("=")[0] for entry in result["invalid_enums"]}
    assert not flagged & set(TYPED)
