"""Prompt builder for the MBPP task."""


def build_prompt(input_data: dict, use_shot: bool = True) -> str:
    """
    Build an MBPP prompt.

    Args:
        input_data: dict with keys ``text`` (description) and ``code`` (reference code).
        use_shot: unused (kept for API consistency).
    """
    func = input_data['prompt']
    code = input_data['code']
    func_name = code.split('def')[-1].split(':')[0]
    prompt = (
        f"Role: You are a professional Python coding assistant\n"
        f"Task: Complete the follow function implementation strictly and clearly "
        f"without any additional comments or explanations.\n"
        f"{func}\n"
        f"the function name and the parameters is {func_name}\n"
    )
    return prompt

