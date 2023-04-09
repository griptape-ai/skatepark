from skatepark.drivers import OpenAiPromptDriver
from skatepark.utils import TiktokenTokenizer
from skatepark.steps import PromptStep, ToolStep, ToolkitStep, Step, ToolSubstep
from skatepark.structures import Pipeline
from skatepark.schemas import PipelineSchema
from skatepark.tools import PingPongTool, CalculatorTool, DataScientistTool, EmailSenderTool, WikiTool


class TestPipelineSchema:
    def test_serialization(self):
        pipeline = Pipeline(
            autoprune_memory=False,
            prompt_driver=OpenAiPromptDriver(
                tokenizer=TiktokenTokenizer(stop_sequence="<test>"),
                temperature=0.12345
            )
        )

        tools = [
            PingPongTool(),
            CalculatorTool(),
            DataScientistTool(),
            EmailSenderTool(host="localhost", port=1025, from_email="test@skateparktest.com", use_ssl=False),
            WikiTool()
        ]

        tool_step = ToolStep("test tool prompt", tool=PingPongTool())

        pipeline.add_steps(
            PromptStep("test prompt"),
            tool_step,
            ToolkitStep("test router step", tools=tools)
        )

        pipeline_dict = PipelineSchema().dump(pipeline)

        assert pipeline_dict["autoprune_memory"] is False
        assert len(pipeline_dict["steps"]) == 3
        assert pipeline_dict["steps"][0]["state"] == "PENDING"
        assert pipeline_dict["steps"][0]["child_ids"][0] == pipeline.steps[1].id
        assert pipeline_dict["steps"][1]["parent_ids"][0] == pipeline.steps[0].id
        assert len(pipeline_dict["steps"][-1]["tools"]) == 5
        assert pipeline_dict["prompt_driver"]["temperature"] == 0.12345
        assert pipeline_dict["prompt_driver"]["tokenizer"]["stop_sequence"] == "<test>"

    def test_deserialization(self):
        pipeline = Pipeline(
            autoprune_memory=False,
            prompt_driver=OpenAiPromptDriver(
                tokenizer=TiktokenTokenizer(stop_sequence="<test>"),
                temperature=0.12345
            )
        )

        tools = [
            PingPongTool(),
            CalculatorTool(),
            DataScientistTool(),
            EmailSenderTool(host="localhost", port=1025, from_email="test@skateparktest.com", use_ssl=False),
            WikiTool()
        ]

        tool_step = ToolStep("test tool prompt", tool=PingPongTool())

        pipeline.add_steps(
            PromptStep("test prompt"),
            tool_step,
            ToolkitStep("test router step", tools=tools)
        )

        workflow_dict = PipelineSchema().dump(pipeline)
        deserialized_pipeline = PipelineSchema().load(workflow_dict)

        assert deserialized_pipeline.autoprune_memory is False
        assert len(deserialized_pipeline.steps) == 3
        assert deserialized_pipeline.steps[0].child_ids[0] == pipeline.steps[1].id
        assert deserialized_pipeline.steps[0].state == Step.State.PENDING
        assert deserialized_pipeline.steps[1].parent_ids[0] == pipeline.steps[0].id
        assert len(deserialized_pipeline.last_step().tools) == 5
        assert deserialized_pipeline.prompt_driver.temperature == 0.12345
        assert deserialized_pipeline.prompt_driver.tokenizer.stop_sequence == "<test>"
