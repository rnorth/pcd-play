"""
Validates the raw text of an autoregressive (naive) JSON generation against a schema.
Shared by the MLX and PyTorch baselines so their schema_match means the same thing.
"""

import json
import re
from typing import Any, Dict

from core.schema import StructuredSchema


def validate_naive_output(raw_text: str, schema: StructuredSchema) -> Dict[str, Any]:
    """
    Extracts the outermost {...} from raw_text and parses it. schema_match requires
    valid JSON, every schema field present and every enum value one of its choices;
    extra keys are allowed. Boolean and number/date/datetime values are not checked.
    """
    cleaned_json_str = raw_text.strip()
    match = re.search(r"(\{.*\})", cleaned_json_str, re.DOTALL)
    if match:
        cleaned_json_str = match.group(1)

    parsed_json = None
    is_valid_json = False
    parse_error = None
    try:
        parsed_json = json.loads(cleaned_json_str)
        is_valid_json = True
    except Exception as e:
        parse_error = str(e)

    missing_keys = []
    invalid_enums = []
    if is_valid_json and isinstance(parsed_json, dict):
        for fname, fdef in schema.fields.items():
            if fname not in parsed_json:
                missing_keys.append(fname)
            elif fdef.field_type != "boolean" and not fdef.is_generative:
                val = str(parsed_json[fname])
                if val not in fdef.choices:
                    invalid_enums.append(f"{fname}={val}")

    return {
        "is_valid_json": is_valid_json,
        "schema_match": is_valid_json and not missing_keys and not invalid_enums,
        "parsed_json": parsed_json,
        "parse_error": parse_error,
        "missing_keys": missing_keys,
        "invalid_enums": invalid_enums,
    }
