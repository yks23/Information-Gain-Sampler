"""Prompt builder for the Creativity Writing task."""


def build_prompt(input_data: dict, use_shot: bool = True) -> str:
    """
    Build a Creativity Writing prompt.

    The instruction (writing prompt) is used directly as the user prompt
    without any additional wrapping or few-shot examples.

    Args:
        input_data: dict with key ``instruction`` (or ``prompt``).
        use_shot: unused (kept for API consistency).
    """
    # Support both 'instruction' and 'prompt' keys
    return input_data.get('instruction', input_data.get('prompt', ''))
