import pytest

from core.schema import FieldDefinition, StructuredSchema


def test_number_field_defaults():
    f = FieldDefinition("amount", "number", "")
    assert f.decimals == 0
    assert f.is_generative
    assert f.choices is None


@pytest.mark.parametrize("decimals", [7, -1])
def test_decimals_out_of_range(decimals):
    with pytest.raises(ValueError):
        FieldDefinition("amount", "number", "", decimals=decimals)


def test_json_schema_prompt_new_types():
    schema = StructuredSchema({
        "amount": {"type": "number"},
        "when": {"type": "date"},
        "at": {"type": "datetime"},
    })
    prompt = schema.to_json_schema_prompt_str()
    assert '"amount": number | null' in prompt
    assert '"when": "YYYY-MM-DD" | null' in prompt
    assert '"at": "YYYY-MM-DDTHH:MM:SS" | null' in prompt
    assert schema.has_generative


def test_generative_suffix_has_no_trailing_space():
    from transformers import AutoTokenizer
    from core.engine_mlx import MODEL_ID

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    schema = StructuredSchema({"amount": {"type": "number"}, "ok": {"type": "boolean"}})
    meta = schema.compile_parallel_metadata(tok)
    assert tok.decode(meta["suffix_tok_lists"][0]) == '  "amount":'
    assert meta["cands_per_field"][0] == []
    assert meta["has_collisions"][0] is False


def test_unknown_type_message_lists_all_types():
    with pytest.raises(ValueError) as e:
        FieldDefinition("x", "string", "")
    for t in ("boolean", "enum", "number", "date", "datetime"):
        assert t in str(e.value)


def test_decimals_must_be_a_real_int():
    with pytest.raises(ValueError):
        FieldDefinition("amount", "number", "", decimals=True)


@pytest.mark.parametrize("field_type", ["date", "datetime", "boolean"])
def test_decimals_only_allowed_on_number(field_type):
    with pytest.raises(ValueError, match="decimals"):
        FieldDefinition("x", field_type, "", decimals=2)


def test_to_dict_includes_decimals():
    assert FieldDefinition("amount", "number", "", decimals=2).to_dict()["decimals"] == 2
    assert FieldDefinition("ok", "boolean", "").to_dict()["decimals"] is None


def test_torch_error_names_typed_fields():
    pytest.importorskip("torch")
    from core.engine_torch import run_parallel_generation_torch

    schema = StructuredSchema({"when": {"type": "date"}, "ok": {"type": "boolean"}})
    with pytest.raises(NotImplementedError, match="'when'"):
        run_parallel_generation_torch("text", schema)


def test_schema_field_cap():
    from core.schema import MAX_FIELDS

    StructuredSchema({f"f{i}": {"type": "boolean"} for i in range(MAX_FIELDS)})
    with pytest.raises(ValueError, match=str(MAX_FIELDS)):
        StructuredSchema({f"f{i}": {"type": "boolean"} for i in range(MAX_FIELDS + 1)})


def test_parallel_catalogue_lists_enum_choices():
    schema = StructuredSchema({
        "kind": {"type": "enum", "description": "Kind of loss", "choices": ["COLLISION", "FIRE"]},
        "hurt": {"type": "boolean", "description": "Injuries"},
    })
    assert schema.to_parallel_schema_str() == (
        '  "kind": Kind of loss [COLLISION | FIRE]\n'
        '  "hurt": Injuries'
    )


def test_parallel_catalogue_omits_choices_of_large_enums():
    choices = [f"C{i}" for i in range(51)]
    schema = StructuredSchema({"code": {"type": "enum", "description": "Code", "choices": choices}})
    assert schema.to_parallel_schema_str() == '  "code": Code'
