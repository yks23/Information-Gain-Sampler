"""Prompt builder for the 4x4 Sudoku task."""

_SYSTEM_PROMPT = """
Solve this 4x4 Sudoku puzzle represented as a 16-digit string (read left-to-right, top-to-bottom) where '0'=empty cell.

Requirements:
1. Replace ALL '0's with digits 1-4
2. Follow STRICT Sudoku rules:
   - Rows: Each must contain 1-4 exactly once
   - Columns: Each must contain 1-4 exactly once
   - 2x2 Boxes: Each must contain 1-4 exactly once
3. Format answer as:
<answer>
[16-digit solution]
</answer>
"""

_FEW_SHOT_PROMPT = """Please solve the following 4x4 Sudoku puzzle. The puzzle is provided as a 16-character string reading left-to-right, top-to-bottom, where '0' represents empty cells.

Here are some examples:


Puzzle: 
4100
0001
1300
2000
<answer> 
4132
3241
1324
2413
</answer>

Puzzle: 
0004
0321
0203
3002
<answer>
2134
4321
1243
3412
</answer>

Puzzle: 
4123
0000
0402
2300
<answer>
4123
3214
1432
2341
</answer>

Puzzle:
1432
0041
3000
4000
<answer>
1432
2341
3214
4123
</answer>

Puzzle:
0020
0341
0210
1002
<answer>
4123
2341
3214
1432
</answer>

Puzzle: {puzzle_str}
<answer>

"""

_NO_SHOT_PROMPT = """Puzzle:
{puzzle_str}
<answer>

"""


def build_prompt(input_data: dict, use_shot: bool = True) -> str:
    """
    Build a Sudoku prompt.

    Args:
        input_data: dict with key ``puzzle`` (a 16-char string).
        use_shot: whether to include few-shot examples.
    """
    puzzle_str = input_data['Puzzle']
    puzzle_str = '\n'.join(puzzle_str[i : i + 4] for i in range(0, len(puzzle_str), 4))
    if use_shot:
        question = _FEW_SHOT_PROMPT.format(puzzle_str=puzzle_str)
        return _SYSTEM_PROMPT + '\n\n' + question
    else:
        return _SYSTEM_PROMPT + '\n\n' + _NO_SHOT_PROMPT.format(puzzle_str=puzzle_str)

