"""
Prompt construction for the autoregressive JSON baseline and the parallel engine.
"""

from core.schema import StructuredSchema


def build_naive_json_prompt(context: str, schema: StructuredSchema) -> str:
    """
    Builds the baseline prompt instructing the model to generate a full JSON document.
    """
    schema_prompt = schema.to_json_schema_prompt_str()
    prompt = (
        f"<|im_start|>system\n"
        f"You are a precise data extraction system. You must output ONLY a valid, beautifully formatted, indented JSON object with newlines and 2-space indentation matching the schema below. Do not output a single-line string. Do not include markdown tags.\n\n"
        f"JSON Schema:\n{schema_prompt}<|im_end|>\n"
        f"<|im_start|>user\n"
        f"Analyze the following context and generate the required formatted JSON object:\n\n{context}<|im_end|>\n"
        f"<|im_start|>assistant\n{{\n  "
    )
    return prompt


def build_parallel_prompt(context: str, schema: StructuredSchema) -> str:
    """
    Builds the shared prefix every parallel field suffix continues from. Schemas with
    number/date/datetime fields also get a null rule; enum/boolean-only prompts don't.
    """
    null_rule = "\nUse null for any attribute the text does not state." if schema.has_generative else ""
    return (
        f"<|im_start|>system\n"
        f"Classify JSON attributes:\n{schema.to_parallel_schema_str()}{null_rule}<|im_end|>\n"
        f"<|im_start|>user\n"
        f"{context}<|im_end|>\n"
        f"<|im_start|>assistant\n{{\n"
    )

