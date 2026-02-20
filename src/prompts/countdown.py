"""Prompt builder for the Countdown task."""


def build_prompt(input_data: dict, use_shot: bool = True) -> str:
    """
    Build a Countdown prompt.

    Args:
        input_data: dict with key ``query`` (e.g. ``"15,44,79,50"``).
        use_shot: whether to include few-shot examples.
    """
    question = input_data['input']
    if use_shot:
        prompt = f'''For the given numbers, find a sequence of arithmetic operations that results in the target number.
Show your reasoning and conclude with "The answer is: [formula]".

Examples:
question: 15,44,79,50
Solution: Let's try to combine 15 and 44. 44 - 15 = 29. Now we have 29 and the remaining number 79. We need to reach the target 50. Let's try 79 - 29 = 50. This works. The answer is: 44-15=29,79-29=50

question: 1,2,12,25
Solution: We have 1, 2, 12 and the target is 25. Let's try multiplying 2 and 12. 2 * 12 = 24. Now we have 24 and the remaining number 1. We need to reach 25. 24 + 1 = 25. This is correct. The answer is: 2*12=24,1+24=25

question: 3,85,5,30
Solution: The numbers are 3, 85, 5 and the target is 30. Let's try adding 85 and 5. 85 + 5 = 90. Now we have 90 and the remaining number 3. We need to reach 30. 90 / 3 = 30. That's the target. The answer is: 85+5=90,90/3=30

New Question: {question}
Solution: '''
    else:
        prompt = f'''For the given numbers, find a sequence of arithmetic operations that results in the target number.
Show your reasoning and conclude with "The answer is: [formula]".

Question: {question}
Solution: '''
    return prompt

