from pydantic import BaseModel
from rich.pretty import pprint

from griptape.rules import Rule
from griptape.structures import Pipeline
from griptape.tasks import PromptTask


class Step(BaseModel):
    explanation: str
    output: str


class Output(BaseModel):
    steps: list[Step]
    final_answer: str


pipeline = Pipeline(
    tasks=[
        PromptTask(
            output_schema=Output,
            rules=[
                Rule("You are a helpful math tutor. Guide the user through the solution step by step."),
            ],
        )
    ]
)

output = pipeline.run("How can I solve 8x + 7 = -23").output.value


pprint(output)  # OutputModel
