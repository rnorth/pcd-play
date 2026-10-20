from core.prompt_builder import build_parallel_prompt
from core.schema import StructuredSchema

NULL_RULE = "Use null for any attribute the text does not state."


def test_parallel_prompt_enum_only_has_no_null_rule():
    schema = StructuredSchema({"ok": {"type": "boolean", "description": "Is it ok"}})
    assert build_parallel_prompt("ctx", schema) == (
        '<|im_start|>system\nClassify JSON attributes:\n  "ok": Is it ok<|im_end|>\n'
        "<|im_start|>user\nctx<|im_end|>\n<|im_start|>assistant\n{\n"
    )


def test_parallel_prompt_typed_fields_get_null_rule():
    schema = StructuredSchema({"when": {"type": "date", "description": "Date"}})
    prompt = build_parallel_prompt("ctx", schema)
    assert f'  "when": Date (YYYY-MM-DD)\n{NULL_RULE}<|im_end|>' in prompt
