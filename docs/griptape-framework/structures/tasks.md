---
search:
  boost: 2
---

## Overview

A [Task](../../reference/griptape/tasks/index.md) is a purpose-built abstraction for the Large Language Model (LLM). Griptape offers various types of Tasks, each suitable for specific use cases.

## Context

Tasks that take input have a field [input](../../reference/griptape/tasks/base_text_input_task.md#griptape.tasks.base_text_input_task.BaseTextInputTask.input) which lets you define the Task objective.
Within the [input](../../reference/griptape/tasks/base_text_input_task.md#griptape.tasks.base_text_input_task.BaseTextInputTask.input), you can access the following [context](../../reference/griptape/structures/structure.md#griptape.structures.structure.Structure.context) variables:

- `args`: an array of arguments passed to the `.run()` method.
- `kwargs`: an array of keyword arguments passed to the `.run()` method.
- `structure`: the structure that the task belongs to.
- user defined context variables

Additional [context](../../reference/griptape/structures/structure.md#griptape.structures.structure.Structure.context) variables may be added based on the Structure running the task.

```python
--8<-- "docs/griptape-framework/structures/src/tasks_1.py"
```

```
[09/08/23 11:12:47] INFO     PromptTask 0f5a5def49864126834627b6140f3e63
                             Input: Respond to the user's following question 'How do I bake a cake?' in the language 'ENGLISH' and tone
                             'PLAYFUL'.
[09/08/23 11:13:17] INFO     PromptTask 0f5a5def49864126834627b6140f3e63
                             Output: Oh, you're in for a treat! Baking a cake is like creating a masterpiece, but way more delicious! Here's a
                             simple recipe to get you started:

                             1. Preheat your oven to 350°F (175°C). It's like sunbathing, but for your cake!

                             2. Grab a bowl and mix together 2 cups of sugar and 1/2 cup of softened butter. It's like making sweet, buttery
                             sandcastles!

                             3. Crack in 3 eggs, one at a time, and stir in 2 teaspoons of vanilla extract. It's a pool party in your bowl!

                             4. In a separate bowl, combine 1 1/2 cups of all-purpose flour, 1 3/4 teaspoons of baking powder, and a pinch of
                             salt. This is the dry gang!

                             5. Gradually mix the dry gang into the buttery pool party. Stir until it's just combined, we don't want to
                             overwork the partygoers!

                             6. Pour the batter into a greased cake pan. It's like tucking your cake into bed!

                             7. Bake for 30 to 40 minutes, or until a toothpick comes out clean. It's like playing hide and seek with your
                             cake!

                             8. Let it cool, then frost and decorate as you like. This is where you can let your creativity shine!

                             Remember, baking is all about having fun and enjoying the process. So, put on your favorite tunes, roll up your
                             sleeves, and let's get baking! 🍰🎉
```

## Hooks

All Tasks implement [RunnableMixin](../../reference/griptape/mixins/runnable_mixin.md) which provides `on_before_run` and `on_after_run` hooks for the Task lifecycle.

These hooks can be used to perform actions before and after the Task is run. For example, you can mask sensitive information before running the Task, and transform the output after the Task is run.

```python
--8<-- "docs/griptape-framework/structures/src/task_hooks.py"
```

```
[10/15/24 15:14:10] INFO     PromptTask 63a0c734059c42808c87dff351adc8ab
                             Input: Respond to this user: Hello! My favorite color is blue, and my social security number is xxx-xx-xxxx.
[10/15/24 15:14:11] INFO     PromptTask 63a0c734059c42808c87dff351adc8ab
                             Output: {
                               "original_input": "Respond to this user: Hello! My favorite color is blue, and my social security number is 123-45-6789.",
                               "masked_input": "Respond to this user: Hello! My favorite color is blue, and my social security number is xxx-xx-xxxx.",
                               "output": "Hello! It's great to hear that your favorite color is blue. However, it's important to keep your personal information, like your
                             social security number, private and secure. If you have any questions or need assistance, feel free to ask!"
                             }
```

## Prompt Task

For general-purpose interaction with LLMs, use the [PromptTask](../../reference/griptape/tasks/prompt_task.md):

```python
--8<-- "docs/griptape-framework/structures/src/tasks_2.py"
```

```
[10/20/23 15:27:26] INFO     PromptTask f5025c6352914e9f80ef730e5269985a        
                             Input: Respond to the following request:     
                             Write me a haiku                                   
[10/20/23 15:27:28] INFO     PromptTask f5025c6352914e9f80ef730e5269985a        
                             Output: Gentle morning dew,                        
                             Kisses the waking flowers,                         
                             Day begins anew.
```

### Tools

You can pass in one or more Tools which the LLM will decide to use through Chain of Thought (CoT) reasoning. Because tool execution uses CoT, it is recommended to only use with very capable models.

```python
--8<-- "docs/griptape-framework/structures/src/tasks_4.py"
```

```
[08/12/24 15:16:30] INFO     PromptTask f5b44fe1dadc4e6688053df71d97e0de
                             Input: Load https://www.griptape.ai, summarize it, and store it in a file called griptape.txt
[08/12/24 15:16:32] INFO     Subtask a4483eddfbe84129b0f4c04ef0f5d695
                             Actions: [
                               {
                                 "tag": "call_AFeOL9MGhZ4mPFCULcBEm4NQ",
                                 "name": "WebScraperTool",
                                 "path": "get_content",
                                 "input": {
                                   "values": {
                                     "url": "https://www.griptape.ai"
                                   }
                                 }
                               }
                             ]
                    INFO     Subtask a4483eddfbe84129b0f4c04ef0f5d695
                             Response: Output of "WebScraperTool.get_content" was stored in memory with memory_name "TaskMemory" and artifact_namespace
                             "c6a6bcfc16f34481a068108aeaa6838e"
[08/12/24 15:16:33] INFO     Subtask ee5f11666ded4dc39b94e4c59d18fbc7
                             Actions: [
                               {
                                 "tag": "call_aT7DX0YSQPmOcnumWXrGoMNt",
                                 "name": "PromptSummaryTool",
                                 "path": "summarize",
                                 "input": {
                                   "values": {
                                     "summary": {
                                       "memory_name": "TaskMemory",
                                       "artifact_namespace": "c6a6bcfc16f34481a068108aeaa6838e"
                                     }
                                   }
                                 }
                               }
                             ]
[08/12/24 15:16:37] INFO     Subtask ee5f11666ded4dc39b94e4c59d18fbc7
                             Response: Output of "PromptSummaryTool.summarize" was stored in memory with memory_name "TaskMemory" and artifact_namespace
                             "669d29a704444176be93d09d014298df"
[08/12/24 15:16:38] INFO     Subtask d9b2dd9f96d841f49f5d460e33905183
                             Actions: [
                               {
                                 "tag": "call_QgMk1M1UuD6DAnxjfQz1MH6X",
                                 "name": "FileManagerTool",
                                 "path": "save_memory_artifacts_to_disk",
                                 "input": {
                                   "values": {
                                     "dir_name": ".",
                                     "file_name": "griptape.txt",
                                     "memory_name": "TaskMemory",
                                     "artifact_namespace": "669d29a704444176be93d09d014298df"
                                   }
                                 }
                               }
                             ]
                    INFO     Subtask d9b2dd9f96d841f49f5d460e33905183
                             Response: Successfully saved memory artifacts to disk
[08/12/24 15:16:39] INFO     PromptTask f5b44fe1dadc4e6688053df71d97e0de
                             Output: The content from https://www.griptape.ai has been summarized and stored in a file called `griptape.txt`.
```

### Images

If the model supports it, you can also pass image inputs:

```python
--8<-- "docs/griptape-framework/structures/src/tasks_3.py"
```

```
[06/21/24 10:01:08] INFO     PromptTask c229d1792da34ab1a7c45768270aada9
                             Input: What's in this image?

                             Media, type: image/jpeg, size: 82351 bytes
[06/21/24 10:01:12] INFO     PromptTask c229d1792da34ab1a7c45768270aada9
                             Output: The image depicts a stunning mountain landscape at sunrise or sunset. The sun is partially visible on the left side of the image,
                             casting a warm golden light over the scene. The mountains are covered with snow at their peaks, and a layer of clouds or fog is settled in the
                             valleys between them. The sky is a mix of warm colors near the horizon, transitioning to cooler blues higher up, with some scattered clouds
                             adding texture to the sky. The overall scene is serene and majestic, highlighting the natural beauty of the mountainous terrain.
```

## Tool Task

Another way to use [Griptape Tools](../../griptape-framework/tools/index.md), is with a [Tool Task](../../reference/griptape/tasks/tool_task.md).
This Task takes in a single Tool which the LLM will use without Chain of Thought (CoT) reasoning. Because this Task does not use CoT, it is better suited for less capable models.

```python
--8<-- "docs/griptape-framework/structures/src/tasks_5.py"
```

```
[10/20/23 14:20:25] INFO     ToolTask df1604b417a84ee781dbd1f2b904ed30          
                             Input: Give me the answer for 5*4.                 
[10/20/23 14:20:29] INFO     Subtask a9a9ad7be2bf465fa82bd350116fabe4           
                             Action: {                                          
                                                                
                               "name": "CalculatorTool",                            
                               "path": "calculate",                         
                               "input": {                                       
                                 "values": {                                    
                                   "expression": "5*4"                          
                                 }                                              
                               }                                                
                             }                                                  
[10/20/23 14:20:30] INFO     Subtask a9a9ad7be2bf465fa82bd350116fabe4           
                             Response: 20                                    
                    INFO     ToolTask df1604b417a84ee781dbd1f2b904ed30          
                             Output: 20       
```

## Extraction Task

To extract information from text, use an [ExtractionTask](../../reference/griptape/tasks/extraction_task.md).
This Task takes an [Extraction Engine](../../griptape-framework/engines/extraction-engines.md), and a set of arguments specific to the Engine.

### CSV Extraction

```python
--8<-- "docs/griptape-framework/structures/src/tasks_6.py"
```

```
[12/19/23 10:33:11] INFO     ExtractionTask e87fb457edf8423ab8a78583badd7a11
                             Input:
                             Alice, 28, lives in New York.
                             Bob, 35 lives in California.
                             Charlie is 40 and lives in Texas.

[12/19/23 10:33:13] INFO     ExtractionTask e87fb457edf8423ab8a78583badd7a11
                             Output: Name,Age,Address
                             Alice,28,New York
                             Bob,35,California
                             Charlie,40,Texas
```

### JSON Extraction

```python
--8<-- "docs/griptape-framework/structures/src/tasks_7.py"
```

```
[12/19/23 10:37:41] INFO     ExtractionTask 3315cc77f94943a2a2dceccfe44f6a67
                             Input:
                             Alice (Age 28) lives in New York.
                             Bob (Age 35) lives in California.

[12/19/23 10:37:44] INFO     ExtractionTask 3315cc77f94943a2a2dceccfe44f6a67
                             Output: {'name': 'Alice', 'age': 28, 'location': 'New York'}
                             {'name': 'Bob', 'age': 35, 'location': 'California'}
```

## Text Summary Task

To summarize a text, use the [TextSummaryTask](../../reference/griptape/tasks/text_summary_task.md).
This Task takes an [Summarization Engine](../../griptape-framework/engines/summary-engines.md), and a set of arguments to the engine.

```python
--8<-- "docs/griptape-framework/structures/src/tasks_8.py"
```

```
[10/20/23 15:37:46] INFO     TextSummaryTask e870f2a6226f43fcb89f93b1c0c85b10   
                             Input: Artificial Intelligence (AI) is a branch of 
                             computer science that deals with creating machines 
                             capable of thinking and learning. It encompasses   
                             various fields such as machine learning, neural    
                             networks, and deep learning. AI has the potential  
                             to revolutionize many sectors, including           
                             healthcare, finance, and transportation. Our life  
                             in this modern age depends largely on computers. It
                             is almost impossible to think about life without   
                             computers. We need computers in everything that we 
                             use in our daily lives. So it becomes very         
                             important to make computers intelligent so that our
                             lives become easy. Artificial Intelligence is the  
                             theory and development of computers, which imitates
                             the human intelligence and senses, such as visual  
                             perception, speech recognition, decision-making,   
                             and translation between languages. Artificial      
                             Intelligence has brought a revolution in the world 
                             of technology.                                     
[10/20/23 15:37:49] INFO     TextSummaryTask e870f2a6226f43fcb89f93b1c0c85b10   
                             Output: Artificial Intelligence (AI) is a branch of
                             computer science that focuses on creating          
                             intelligent machines. It encompasses various fields
                             such as machine learning and neural networks. AI   
                             has the potential to revolutionize sectors like    
                             healthcare, finance, and transportation. It is     
                             essential to make computers intelligent to simplify
                             our daily lives. AI imitates human intelligence and
                             senses, bringing a revolution in technology.   
```

## RAG Task

To query text, use the [RagTask](../../reference/griptape/tasks/rag_task.md).
This task takes a [RAG Engine](../../griptape-framework/engines/rag-engines.md), and a set of arguments specific to the engine.

```python
--8<-- "docs/griptape-framework/structures/src/tasks_9.py"
```

## Code Execution Task

To execute an arbitrary Python function, use the [CodeExecutionTask](../../reference/griptape/tasks/code_execution_task.md).
This task takes a python function, and authors can elect to return a custom artifact.

```python
--8<-- "docs/griptape-framework/structures/src/tasks_10.py"
```

```
[01/09/24 15:23:54] INFO     CodeExecutionTask 048b1f548683475187064dde90055f72 
                             Input: Write me a line in a poem                   
                    INFO     CodeExecutionTask 048b1f548683475187064dde90055f72 
                             Output: 25                                         
                    INFO     PromptTask b6156dc5c0c6404488ab925989e78b01        
                             Input: Write me a line in a poem using 25          
                             characters                                         
[01/09/24 15:24:03] INFO     PromptTask b6156dc5c0c6404488ab925989e78b01        
                             Output: "Silent code, loud impact."  
```

## Branch Task

By default, a [Workflow](../structures/workflows.md) will only run a Task when all the Tasks it depends on have finished.

You can make use of [BranchTask](../../reference/griptape/tasks/branch_task.md) in order to tell the Workflow not to run all dependent tasks, but instead to pick and choose one or more paths to go down.

The `BranchTask`'s [on_run](../../reference/griptape/tasks/branch_task.md#griptape.tasks.branch_task.BranchTask.on_run) function can return one of three things:

1. An `InfoArtifact` containing the `id` of the Task to run next.
1. A `ListArtifact` of `InfoArtifact`s containing the `id`s of the Tasks to run next.
1. An _empty_ `ListArtifact` to indicate that no Tasks should be run next.

```python
--8<-- "docs/griptape-framework/structures/src/branch_task.py"
```

## Image Generation Tasks

To generate an image, use one of the following [Image Generation Tasks](../../reference/griptape/tasks/index.md). All Image Generation Tasks accept an [Image Generation Driver](../drivers/image-generation-drivers.md).

All successful Image Generation Tasks will always output an [Image Artifact](../data/artifacts.md#image). Each task can be configured to additionally write the generated image to disk by providing either the `output_file` or `output_dir` field. The `output_file` field supports file names in the current directory (`my_image.png`), relative directory prefixes (`images/my_image.png`), or absolute paths (`/usr/var/my_image.png`). By setting `output_dir`, the task will generate a file name and place the image in the requested directory.

### Prompt Image Generation Task

The [Prompt Image Generation Task](../../reference/griptape/tasks/prompt_image_generation_task.md) generates an image from a text prompt.

```python
--8<-- "docs/griptape-framework/structures/src/tasks_11.py"
```

### Variation Image Generation Task

The [Variation Image Generation Task](../../reference/griptape/tasks/variation_image_generation_task.md) generates an image using an input image and a text prompt. The input image is used as a basis for generating a new image as requested by the text prompt.

```python
--8<-- "docs/griptape-framework/structures/src/tasks_12.py"
```

### Inpainting Image Generation Task

The [Inpainting Image Generation Task](../../reference/griptape/tasks/inpainting_image_generation_task.md) generates an image using an input image, a mask image, and a text prompt. The input image will be modified within the bounds of the mask image as requested by the text prompt.

```python
--8<-- "docs/griptape-framework/structures/src/tasks_13.py"
```

### Outpainting Image Generation Task

The [Outpainting Image Generation Task](../../reference/griptape/tasks/outpainting_image_generation_task.md) generates an image using an input image, a mask image, and a text prompt. The input image will be modified outside the bounds of a mask image as requested by the text prompt.

```python
--8<-- "docs/griptape-framework/structures/src/tasks_14.py"
```

## Structure Run Task

The [Structure Run Task](../../reference/griptape/tasks/structure_run_task.md) runs another Structure with a given input.
This Task is useful for orchestrating multiple specialized Structures in a single run. Note that the input to the Task is a tuple of arguments that will be passed to the Structure.

```python
--8<-- "docs/griptape-framework/structures/src/tasks_16.py"
```

## Assistant Task

The [Assistant Task](../../reference/griptape/tasks/assistant_task.md) enables Structures to interact with various "assistant" services using [Assistant Drivers](../../reference/griptape/drivers/assistant/index.md).

```python
--8<-- "docs/griptape-framework/structures/src/tasks_assistant.py"
```

## Text to Speech Task

This Task enables Structures to synthesize speech from text using [Text to Speech Drivers](../../reference/griptape/drivers/text_to_speech/index.md).

```python
--8<-- "docs/griptape-framework/structures/src/tasks_17.py"
```

## Audio Transcription Task

This Task enables Structures to transcribe speech from text using [Audio Transcription Drivers](../../reference/griptape/drivers/audio_transcription/index.md).

```python
--8<-- "docs/griptape-framework/structures/src/tasks_18.py"
```
