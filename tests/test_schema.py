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
