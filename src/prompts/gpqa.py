"""Prompt builder for the GPQA task."""

_SHOT1 = '''Here are some example questions from experts. Answer the final question yourself, following the format of the previous questions exactly.
Question: A large gene has dozens of exons, of which the central ones code for folded triple helical repeats that connect the cytoskeleton with sarcolemma and extracellular space. Each exon usually codes for one folded triple alpha helix. The most common mutations of the gene are central exon deletions that create out-of-frame peptides and progressive degenerative organ waste. A solution is to deliver a Morpholino that recognizes the 5' end of the out-of-frame exon in pre-mRNA. The molecule prevents binding of the spliceosome and creates exon skipping and in-frame joining. Several missing exons are well tolerated by an organism. Which structure below is not involved in the proposed therapy?
Choices:
(A) lariat
(B) polyA tail
(C) antisense
(D) R-loops
Thought Process: The text describes the dystrophin gene and the FDA-approved oligonucleotide therapy that causes exon skipping by creating a functional, albeit shorter, dystrophin protein. Morpholino is bound to the pre-mRNA in an antisense orientation. Every splicing mechanism creates the lariat molecule that is circular with a 3' tail and soon degraded. The spliced RNA is polyadenylated at the 3' end. R-loops are triple helix of DNA and the pre-mRNA and a consequence of the RNA transcription, not splicing and RNA maturation. The answer is (D)'''

_SHOT2 = '''Question: Two quantum states with energies E1 and E2 have a lifetime of 10^-9 sec and 10^-8 sec, respectively. We want to clearly distinguish these two energy levels. Which one of the following options could be their energy difference so that they can be clearly resolved?
Choices:
(A) 10^-11 eV
(B) 10^-8 eV
(C) 10^-9 eV
(D) 10^-4 eV
Thought Process: According to the uncertainty principle, Delta E* Delta t=hbar/2. Delta t is the lifetime and Delta E is the width of the energy level. With Delta t=10^-9 s==> Delta E1= 3.3 10^-7 ev. And Delta t=10^-11 s gives Delta E2=3.310^-8 eV. Therefore, the energy difference between the two states must be significantly greater than 10^-7 ev. So the answer is 10^-4 ev. The answer is (D)'''

_SHOT3 = '''Question: trans - cinnamaldehyde was treated with methylmagnesium bromide, forming product 1.
1 was treated with pyridinium chlorochromate, forming product 2.
3 was treated with (dimethyl(oxo)-l6 - sulfaneylidene)methane in DMSO at elevated temperature, forming product 3.
how many carbon atoms are there in product 3?
Choices:
(A) 10
(B) 11
(C) 12
(D) 14
Thought Process: trans - cinnamaldehyde was treated with methylmagnesium bromide, forming (E)-4 - phenylbut - 3 - en - 2 - ol. (E)-4 - phenylbut - 3 - en - 2 - ol was treated with pyridinium chlorochromate, forming (E)-4 - phenylbut - 3 - en - 2 - one. (E)-4 - phenylbut - 3 - en - 2 - one was treated with (dimethyl(oxo)-l6 - sulfaneylidene)methane in DMSO at elevated temperature, forming 1-(2 - phenylcyclopropyl)ethan - 1 - one. Chemical Formula: C11H12O. The answer is (B)'''

_SHOT4 = '''Question: how many of the following compounds exhibit optical activity?
1 - methyl - 4-(prop - 1 - en - 2 - yl)cyclohex - 1 - ene
2,3,3,3 - tetrafluoroprop - 1 - ene
di(cyclohex - 2 - en - 1 - ylidene)methane
5-(5 - methylhexan - 2 - ylidene)cyclopenta - 1,3 - diene
3-(2 - methylbut - 1 - en - 1 - ylidene)cyclohex - 1 - ene
[1,1'-biphenyl] - 3,3'-diol
8,8 - dichlorobicyclo[4.2.0]octan - 7 - one
cyclopent - 2 - en - 1 - one
Choices:
(A) 3
(B) 4
(C) 5
(D) 6
Thought Process: the compounds 1 - methyl - 4-(prop - 1 - en - 2 - yl)cyclohex - 1 - ene, 3-(2 - methylbut - 1 - en - 1 - ylidene)cyclohex - 1 - ene, di(cyclohex - 2 - en - 1 - ylidene)methane, 8,8 - dichlorobicyclo[4.2.0]octan - 7 - one are chiral molecules, and thus will be optically active. all the others have a mirror plane of symmetry, and will be achiral. The answer is (B)'''

_SHOT5 = '''Question: A coating is applied to a substrate resulting in a perfectly smooth surface. The measured contact angles of this smooth coating are 132° and 102° for water and hexadecane respectively. The coating formulation is then modified and when now applied to the same type of substrate, a rough surface is produced. When a droplet of water or oil sits on the rough surface, the wettability of the surface can now be described by the Cassie - Baxter state. The water contact angle on the rough surface is now 148°. What would be the best estimate of the contact angle of a droplet of octane on the rough surface? 
Choices:
(A) 129°
(B) 134°
(C) 139°
(D) 124°
Thought Process: In the Cassie - Baxter state, droplets are in contact with a non - uniform surface: some of the droplet is in contact with the coating and some with air. The contact angle (θCB) of a droplet in the Cassie - Baxter state is given by: cosθCB = f1.cosθ1 + f2.cosθ2. Where f1 and f2 are the area fractions of the two components of the surface, in this case coating (f1) and air (f2). θ1 is the contact angle if the droplet was purely in contact with the coating, and θ2 is the contact angle if the droplet was purely in contact with air. First we need to calculate the f1 and f2 using the data we are given for water. We have θCB = 148°, θ1 = 132°, and θ2 is taken to be 180° (contact angle on air). We then have cos(148) = f1.cos(132) + f2.cos(180). By using f1 + f2 = 1, we can solve to give f1 = 0.46 and f2 = 0.54. Next we need to calculate the contact angle of hexadecane on the rough surface, we have θ1 = 102°, f1 = 0.46, f2 = 0.54, and θ2 is taken to be 180° (contact angle on air). Therefore, θCB = 129° for hexadecane. The question however asks about a droplet of octane. Octane is a shorter oil molecule than hexadecane and has a lower surface tension than hexadecane. For a given surface, the contact angle of octane is therefore always lower than for hexadecane. Therefore, the answer is 124° as this is the only answer lower than the 129° of hexadecane.  The answer is (D)'''


def build_prompt(input_data: dict, use_shot: bool = True) -> str:
    """
    Build a GPQA prompt.

    Args:
        input_data: dict with keys ``Question``, ``Correct Answer``, ``Incorrect Answer 1``,
                    ``Incorrect Answer 2``, ``Incorrect Answer 3``.
        use_shot: whether to include few-shot examples.
    """
    question = input_data['question']
    choice1 = input_data['correct_answer']
    choice2 = input_data['option_A']
    choice3 = input_data['option_B']
    choice4 = input_data['option_C']

    question_part = f'''Question: {question}
Choices:
(A) {choice1}
(B) {choice2}
(C) {choice3}
(D) {choice4}
Thought Process: '''

    if use_shot:
        return (
            _SHOT1 + "\n\n" + _SHOT2 + "\n\n" + _SHOT3 + "\n\n"
            + _SHOT4 + "\n\n" + _SHOT5 + "\n\n" + question_part
        )
    else:
        return question_part

