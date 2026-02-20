"""Prompt builder for the MATH-500 task."""

_FEW_SHOT_TEMPLATE = r'''Q: Let \[f(x) = \left\{
\begin{array}{cl} ax+3, &\text{ if }x>2, \\
x-5 &\text{ if } -2 \le x \le 2, \\
2x-b &\text{ if } x <-2.
\end{array}
\right.\]Find $a+b$ if the piecewise function is continuous (which means that its graph can be drawn without lifting your pencil from the paper).

  A: <think>
For the piecewise function to be continuous, the cases must "meet" at $2$ and $-2$. For example, $ax+3$ and $x-5$ must be equal when $x=2$. This implies $a(2)+3=2-5$, which we solve to get $2a=-6 \Rightarrow a=-3$. Similarly, $x-5$ and $2x-b$ must be equal when $x=-2$. Substituting, we get $-2-5=2(-2)-b$, which implies $b=3$. So $a+b=-3+3=\boxed{0}$.
</think>
<answer>
0
</answer>

Q: A rectangular band formation is a formation with $m$ band members in each of $r$ rows, where $m$ and $r$ are integers. A particular band has less than 100 band members. The director arranges them in a rectangular formation and finds that he has two members left over. If he increases the number of members in each row by 1 and reduces the number of rows by 2, there are exactly enough places in the new formation for each band member. What is the largest number of members the band could have?

  A: <think>
Let $x$ be the number of band members in each row for the original formation, when two are left over. Then we can write two equations from the given information: $$rx+2=m$$ $$(r-2)(x+1)=m$$ Setting these equal, we find: $$rx+2=(r-2)(x+1)=rx-2x+r-2$$ $$2=-2x+r-2$$ $$4=r-2x$$ We know that the band has less than 100 members. Based on the first equation, we must have $rx$ less than 98. We can guess and check some values of $r$ and $x$ in the last equation. If $r=18$, then $x=7$, and $rx=126$ which is too big. If $r=16$, then $x=6$, and $rx=96$, which is less than 98. Checking back in the second formation, we see that $(16-2)(6+1)=14\cdot 7=98$ as it should. This is the best we can do, so the largest number of members the band could have is $\boxed{98}$.
</think>
<answer>
98
</answer>

Q: What is the degree of the polynomial $(4 +5x^3 +100 +2\pi x^4 + \sqrt{10}x^4 +9)$?

  A: <think>
This polynomial is not written in standard form. However, we don't need to write it in standard form, nor do we need to pay attention to the coefficients. We just look for the exponents on $x$. We have an $x^4$ term and no other term of higher degree, so $\boxed{4}$ is the degree of the polynomial.
</think>
<answer>
4
</answer>

Q: Evaluate $\left\lceil3\left(6-\frac12\right)\right\rceil$.

  A: <think>
Firstly, $3\left(6-\frac12\right)=18-1-\frac12=17-\frac12$. Because $0\le\frac12<1$, we have $\left\lceil17-\frac12\right\rceil=\boxed{17}$.
</think>
<answer>
17
</answer>

Q: {{question}}

  A:
'''

_ZERO_SHOT_TEMPLATE = r'''Q: {{question}}

  A:
'''


def build_prompt(input_data: dict, use_shot: bool = True) -> str:
    """
    Build a MATH-500 prompt.

    Args:
        input_data: dict with key ``problem``.
        use_shot: whether to include few-shot examples.
    """
    question = input_data['problem']
    template = _FEW_SHOT_TEMPLATE if use_shot else _ZERO_SHOT_TEMPLATE
    return template.replace("{{question}}", question)

