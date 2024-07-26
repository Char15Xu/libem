from libem.core.struct import Prompt, Shots
from libem.core.struct.prompt import (
    Shot, Rules, Experiences
)

"""System prompts"""
role = Prompt(
    default="You are an entity matcher that determines whether "
            "two entity descriptions refer to the same real-world entity.",
    options=[""],
)

rules = Prompt(
    default=Rules(),
    options=[],
)

experiences = Prompt(
    default=Experiences(),
    options=[],
)

output = Prompt(
    default="At the end, give your answer in the form of a single 'yes' or 'no'.",
    options=["Do not provide explanation. Please only give your answer in the form of a single 'yes' or 'no'."],
)

"""Assistant prompts"""
shots = Shots(
    default=[Shot()]
)

"""User prompts"""
query = Prompt(
    default="Entity 1: {left}.\nEntity 2: {right}.",
    options=[],
)
