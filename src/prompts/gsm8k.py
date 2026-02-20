"""Prompt builder for the GSM8K task."""


def build_prompt(input_data: dict, use_shot: bool = True) -> str:
    """
    Build a GSM8K prompt.

    Args:
        input_data: dict with key ``question``.
        use_shot: whether to include few-shot examples.
    """
    question = input_data['question']
    if use_shot:
        prompt = f'''Please solve the new question step by step just like the following examples. For this question:
1. Break down the problem into logical steps
2. Show all intermediate calculations
3. Conclude with "So the answer is..." format

Examples:
question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
target: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6.
question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
target: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.

New Question: {question}
Solution: '''
    else:
        prompt = f'''Please solve the question step by step. For this question:
1. Break down the problem into logical steps
2. Show all intermediate calculations
3. Conclude with "So the answer is..." format

Question: {question}
Solution: '''
    return prompt

