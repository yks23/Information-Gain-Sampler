"""Prompt builder for the HumanEval task."""


def build_prompt(input_data: dict, use_shot: bool = True) -> str:
    """
    Build a HumanEval prompt.

    Args:
        input_data: dict with key ``prompt`` containing the function stub.
        use_shot: unused (kept for API consistency).
    """
    func = input_data['prompt']
    prompt = (
        f"Role: You are a professional Python coding assistant\n"
        f"Task: Complete the follow function implementation strictly and clearly "
        f"without any additional comments or explanations.\n"
        f"{func}"
    )
    return prompt

