---
search:
  boost: 2
---

## Overview

Prompt Drivers are used by Griptape Structures to make API calls to the underlying LLMs. [OpenAi Chat](#openai-chat) is the default prompt driver used in all structures.

You can instantiate drivers and pass them to structures:

```python
--8<-- "docs/griptape-framework/drivers/src/prompt_drivers_1.py"
```

Or use them independently:

```python
--8<-- "docs/griptape-framework/drivers/src/prompt_drivers_2.py"
```

You can pass images to the Driver if the model supports it:

```python
--8<-- "docs/griptape-framework/drivers/src/prompt_drivers_images.py"
```

## Structured Output

Some LLMs provide functionality often referred to as "Structured Output".
This means instructing the LLM to output data in a particular format, usually JSON.
This can be useful for forcing the LLM to output in a parsable format that can be used by downstream systems.

!!! warning

    Each Driver may have a different default setting depending on the LLM provider's capabilities.

### Prompt Task

The easiest way to get started with structured output is by using a [PromptTask](../structures/tasks.md#prompt)'s [output_schema](../../reference/griptape/tasks/prompt_task.md#griptape.tasks.PromptTask.output_schema) parameter.

You can change _how_ the output is structured by setting the Driver's [structured_output_strategy](../../reference/griptape/drivers/prompt/base_prompt_driver.md#griptape.drivers.prompt.base_prompt_driver.BasePromptDriver.structured_output_strategy) to one of:

- `native`: The Driver will use the LLM's structured output functionality provided by the API.
- `tool`: The Driver will add a special tool, [StructuredOutputTool](../../reference/griptape/tools/structured_output/tool.md), and will try to force the LLM to use the Tool.
- `rule`: The Driver will add a [JsonSchemaRule](../structures/rulesets.md#json-schema-rule) to the Task's system prompt. This strategy does not guarantee that the LLM will output JSON and should only be used as a last resort.

You can specify `output_schema` using either `pydantic` or `schema` libraries, though `pydantic` is recommended.

=== "Pydantic"

    ```python
    --8<-- "docs/griptape-framework/drivers/src/prompt_drivers_structured_output_pydantic.py"
    ```

=== "Schema"

    ```python
    --8<-- "docs/griptape-framework/drivers/src/prompt_drivers_structured_output_schema.py"
    ```

## Prompt Drivers

Griptape offers the following Prompt Drivers for interacting with LLMs.

### OpenAI Chat

The [OpenAiChatPromptDriver](../../reference/griptape/drivers/prompt/openai_chat_prompt_driver.md) connects to the [OpenAI Chat](https://platform.openai.com/docs/guides/chat) API.
This driver uses [OpenAi function calling](https://platform.openai.com/docs/guides/function-calling) when using [Tools](../tools/index.md).

```python
--8<-- "docs/griptape-framework/drivers/src/prompt_drivers_3.py"
```

!!! info

    `response_format` and `seed` are unique to the OpenAI Chat Prompt Driver and Azure OpenAi Chat Prompt Driver.

### OpenAI Compatible

Many services such as [LMStudio](https://lmstudio.ai/) and [OhMyGPT](https://www.ohmygpt.com/) provide OpenAI-compatible APIs. You can use the [OpenAiChatPromptDriver](../../reference/griptape/drivers/prompt/openai_chat_prompt_driver.md) to interact with these services.
Simply set the `base_url` to the service's API endpoint and the `model` to the model name. If the service requires an API key, you can set it in the `api_key` field.

```python
--8<-- "docs/griptape-framework/drivers/src/prompt_drivers_4.py"
```

!!! tip

    Make sure to include `v1` at the end of the `base_url` to match the OpenAI API endpoint.

### Azure OpenAI Chat

The [AzureOpenAiChatPromptDriver](../../reference/griptape/drivers/prompt/azure_openai_chat_prompt_driver.md) connects to Azure OpenAI [Chat Completion](https://learn.microsoft.com/en-us/azure/cognitive-services/openai/reference) APIs.
This driver uses [Azure OpenAi function calling](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/function-calling) when using [Tools](../tools/index.md).

```python
--8<-- "docs/griptape-framework/drivers/src/prompt_drivers_5.py"
```

### Cohere

The [CoherePromptDriver](../../reference/griptape/drivers/prompt/cohere_prompt_driver.md) connects to the Cohere [Chat](https://docs.cohere.com/docs/chat-api) API.
This driver uses [Cohere tool use](https://docs.cohere.com/docs/tools) when using [Tools](../tools/index.md).

!!! info

    This driver requires the `drivers-prompt-cohere` [extra](../index.md#extras).

```python
--8<-- "docs/griptape-framework/drivers/src/prompt_drivers_6.py"
```

### Anthropic

!!! info

    This driver requires the `drivers-prompt-anthropic` [extra](../index.md#extras).

The [AnthropicPromptDriver](../../reference/griptape/drivers/prompt/anthropic_prompt_driver.md) connects to the Anthropic [Messages](https://docs.anthropic.com/claude/reference/messages_post) API.
This driver uses [Anthropic tool use](https://docs.anthropic.com/en/docs/build-with-claude/tool-use) when using [Tools](../tools/index.md).

```python
--8<-- "docs/griptape-framework/drivers/src/prompt_drivers_7.py"
```

### Google

!!! info

    This driver requires the `drivers-prompt-google` [extra](../index.md#extras).

The [GooglePromptDriver](../../reference/griptape/drivers/prompt/google_prompt_driver.md) connects to the [Google Generative AI](https://ai.google.dev/tutorials/python_quickstart#generate_text_from_text_inputs) API.
This driver uses [Gemini function calling](https://ai.google.dev/gemini-api/docs/function-calling) when using [Tools](../tools/index.md).

```python
--8<-- "docs/griptape-framework/drivers/src/prompt_drivers_8.py"
```

### Amazon Bedrock

!!! info

    This driver requires the `drivers-prompt-amazon-bedrock` [extra](../index.md#extras).

The [AmazonBedrockPromptDriver](../../reference/griptape/drivers/prompt/amazon_bedrock_prompt_driver.md) uses [Amazon Bedrock](https://aws.amazon.com/bedrock/)'s [Converse API](https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html).
This driver uses [Bedrock tool use](https://docs.aws.amazon.com/bedrock/latest/userguide/tool-use.html) when using [Tools](../tools/index.md).

All models supported by the Converse API are available for use with this driver.

```python
--8<-- "docs/griptape-framework/drivers/src/prompt_drivers_9.py"
```

### Ollama

!!! info

    This driver requires the `drivers-prompt-ollama` [extra](../index.md#extras).

The [OllamaPromptDriver](../../reference/griptape/drivers/prompt/ollama_prompt_driver.md) connects to the [Ollama Chat Completion API](https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-chat-completion).
This driver uses [Ollama tool calling](https://ollama.com/blog/tool-support) when using [Tools](../tools/index.md).

```python
--8<-- "docs/griptape-framework/drivers/src/prompt_drivers_10.py"
```

### Hugging Face Hub

!!! info

    This driver requires the `drivers-prompt-huggingface` [extra](../index.md#extras).

The [HuggingFaceHubPromptDriver](../../reference/griptape/drivers/prompt/huggingface_hub_prompt_driver.md) connects to the [Hugging Face Hub API](https://huggingface.co/docs/hub/api).

!!! warning

    Not all models featured on the Hugging Face Hub are supported by this driver. Models that are not supported by
    [Hugging Face serverless inference](https://huggingface.co/docs/api-inference/en/index) will not work with this driver.
    Due to the limitations of Hugging Face serverless inference, only models that are than 10GB are supported.

```python
--8<-- "docs/griptape-framework/drivers/src/prompt_drivers_11.py"
```

#### Text Generation Interface

The [HuggingFaceHubPromptDriver](#hugging-face-hub) also supports [Text Generation Interface](https://huggingface.co/docs/text-generation-inference/basic_tutorials/consuming_tgi#inference-client) for running models locally. To use Text Generation Interface, just set `model` to a TGI endpoint.

```python
--8<-- "docs/griptape-framework/drivers/src/prompt_drivers_12.py"
```

### Hugging Face Pipeline

!!! info

    This driver requires the `drivers-prompt-huggingface-pipeline` [extra](../index.md#extras).

The [HuggingFacePipelinePromptDriver](../../reference/griptape/drivers/prompt/huggingface_pipeline_prompt_driver.md) uses [Hugging Face Pipelines](https://huggingface.co/docs/transformers/main_classes/pipelines) for inference locally.

!!! warning

    Running a model locally can be a computationally expensive process.

```python
--8<-- "docs/griptape-framework/drivers/src/prompt_drivers_13.py"
```

### Amazon SageMaker Jumpstart

!!! info

    This driver requires the `drivers-prompt-amazon-sagemaker` [extra](../index.md#extras).

The [AmazonSageMakerJumpstartPromptDriver](../../reference/griptape/drivers/prompt/amazon_sagemaker_jumpstart_prompt_driver.md) uses [Amazon SageMaker Jumpstart](https://docs.aws.amazon.com/sagemaker/latest/dg/studio-jumpstart.html) for inference on AWS.

Amazon Sagemaker Jumpstart provides a wide range of models with varying capabilities.
This Driver has been primarily _chat-optimized_ models that have a [Huggingface Chat Template](https://huggingface.co/docs/transformers/en/chat_templating) available.
If your model does not fit this use-case, we suggest sub-classing [AmazonSageMakerJumpstartPromptDriver](../../reference/griptape/drivers/prompt/amazon_sagemaker_jumpstart_prompt_driver.md) and overriding the `_to_model_input` and `_to_model_params` methods.

```python
--8<-- "docs/griptape-framework/drivers/src/prompt_drivers_14.py"
```
