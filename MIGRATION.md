# Migration Guide

This document provides instructions for migrating your codebase to accommodate breaking changes introduced in new versions of Griptape.

## 0.33.X to 0.34.X

### `EventListener.handler` behavior, `driver` parameter rename

Returning `None` from the `handler` function now causes the event to not be published to the `EventListenerDriver`.
The `handler` function can now return a `BaseEvent` object.

#### Before

```python
def handler_fn_return_none(event: BaseEvent) -> Optional[dict]:
    # This causes the `BaseEvent` object to be passed to the EventListenerDriver
    return None

def handler_fn_return_dict(event: BaseEvent) -> Optional[dict]:
    # This causes the returned dictionary to be passed to the EventListenerDriver
    return {
        "key": "value
    }

EventListener(handler=handler_fn_return_none, driver=driver)
EventListener(handler=handler_fn_return_dict, driver=driver)
```

#### After

```python
def handler_fn_return_none(event: BaseEvent) -> Optional[dict | BaseEvent]:
    # This causes the `BaseEvent` object to NOT get passed to the EventListenerDriver
    return None

def handler_fn_return_dict(event: BaseEvent) -> Optional[dict | BaseEvent]:
    # This causes the returned dictionary to be passed to the EventListenerDriver
    return {
        "key": "value
    }

def handler_fn_return_base_event(event: BaseEvent) -> Optional[dict | BaseEvent]:
    # This causes the returned `BaseEvent` object to be passed to the EventListenerDriver
    return ChildClassOfBaseEvent()

# `driver` has been renamed to `event_listener_driver`
EventListener(handler=handler_fn_return_none, event_listener_driver=driver)
EventListener(handler=handler_fn_return_dict, event_listener_driver=driver)
EventListener(handler=handler_fn_return_base_event, event_listener_driver=driver)
```

### Changed `RestApiTool` fields

The following fields are no longer stringified full schemas. They are now just the properties of the schema.

- `RestApiTool.request_path_params_schema` (`list`)
- `RestApiTool.request_query_params_schema` (`dict`)
- `RestApiTool.request_body_schema` (`dict`)

#### Before

```python
posts_client = RestApiTool(
    base_url="https://jsonplaceholder.typicode.com",
    path="posts",
    description="Allows for creating, updating, deleting, patching, and getting posts.",
    request_body_schema=dumps(
        {
            "$schema": "https://json-schema.org/draft/2019-09/schema",
            "$id": "http://example.com/example.json",
            "type": "object",
            "default": {},
            "title": "Root Schema",
            "required": ["title", "body", "userId"],
            "properties": {
                "title": {
                    "type": "string",
                    "default": "",
                    "title": "The title Schema",
                },
                "body": {
                    "type": "string",
                    "default": "",
                    "title": "The body Schema",
                },
                "userId": {
                    "type": "integer",
                    "default": 0,
                    "title": "The userId Schema",
                },
            },
        }
    ),
    request_query_params_schema=dumps(
        {
            "$schema": "https://json-schema.org/draft/2019-09/schema",
            "$id": "http://example.com/example.json",
            "type": "object",
            "default": {},
            "title": "Root Schema",
            "required": ["userId"],
            "properties": {
                "userId": {
                    "type": "string",
                    "default": "",
                    "title": "The userId Schema",
                },
            },
        }
    ),
    response_body_schema=dumps(
        {
            "$schema": "https://json-schema.org/draft/2019-09/schema",
            "$id": "http://example.com/example.json",
            "type": "object",
            "default": {},
            "title": "Root Schema",
            "required": ["id", "title", "body", "userId"],
            "properties": {
                "id": {
                    "type": "integer",
                    "default": 0,
                    "title": "The id Schema",
                },
                "title": {
                    "type": "string",
                    "default": "",
                    "title": "The title Schema",
                },
                "body": {
                    "type": "string",
                    "default": "",
                    "title": "The body Schema",
                },
                "userId": {
                    "type": "integer",
                    "default": 0,
                    "title": "The userId Schema",
                },
            },
        }
    ),
)
```

#### After

```python
posts_client = RestApiTool(
    base_url="https://jsonplaceholder.typicode.com",
    path="posts",
    description="Allows for creating, updating, deleting, patching, and getting posts.",
    request_body_schema={
        "title": str,
        "body": str,
        "userId": int,
    },
    request_query_params_schema={
        "userId": str,
    },
)
```

## 0.32.X to 0.33.X

### Removed `DataframeLoader`

`DataframeLoader` has been removed. Use `CsvLoader.parse` or build `TextArtifact`s from the dataframe instead.

#### Before

```python
DataframeLoader().load(df)
```

#### After

```python
# Convert the dataframe to csv bytes and parse it
CsvLoader().parse(bytes(df.to_csv(line_terminator='\r\n', index=False), encoding='utf-8'))
# Or build TextArtifacts from the dataframe
[TextArtifact(row) for row in source.to_dict(orient="records")]
```

### `TextLoader`, `PdfLoader`, `ImageLoader`, and `AudioLoader` now take a `str | PathLike` instead of `bytes`.

#### Before

```python
PdfLoader().load(Path("attention.pdf").read_bytes())
PdfLoader().load_collection([Path("attention.pdf").read_bytes(), Path("CoT.pdf").read_bytes()])
```

#### After

```python
PdfLoader().load("attention.pdf")
PdfLoader().load_collection([Path("attention.pdf"), "CoT.pdf"])
```

### Removed `fileutils.load_file` and `fileutils.load_files`

`griptape.utils.file_utils.load_file` and `griptape.utils.file_utils.load_files` have been removed.
You can now pass the file path directly to the Loader.

#### Before

```python
PdfLoader().load(load_file("attention.pdf").read_bytes())
PdfLoader().load_collection(list(load_files(["attention.pdf", "CoT.pdf"]).values()))
```

```python
PdfLoader().load("attention.pdf")
PdfLoader().load_collection(["attention.pdf", "CoT.pdf"])
```

### Loaders no longer chunk data

Loaders no longer chunk the data after loading it. If you need to chunk the data, use a [Chunker](https://docs.griptape.ai/stable/griptape-framework/data/chunkers/) after loading the data.

#### Before

```python
chunks = PdfLoader().load("attention.pdf")
vector_store.upsert_text_artifacts(
    {
        "griptape": chunks,
    }
)
```

#### After

```python
artifact = PdfLoader().load("attention.pdf")
chunks = Chunker().chunk(artifact)
vector_store.upsert_text_artifacts(
    {
        "griptape": chunks,
    }
)
```

### Removed `torch` extra from `transformers` dependency

The `torch` extra has been removed from the `transformers` dependency. If you require `torch`, install it separately.

#### Before

```bash
pip install griptape[drivers-prompt-huggingface-hub]
```

#### After

```bash
pip install griptape[drivers-prompt-huggingface-hub]
pip install torch
```

### Removed `MediaArtifact`

`MediaArtifact` has been removed. Use `ImageArtifact` or `AudioArtifact` instead.

#### Before

```python
image_media = MediaArtifact(
    b"image_data",
    media_type="image",
    format="jpeg"
)

audio_media = MediaArtifact(
    b"audio_data",
    media_type="audio",
    format="wav"
)
```

#### After

```python
image_artifact = ImageArtifact(
    b"image_data",
    format="jpeg"
)

audio_artifact = AudioArtifact(
    b"audio_data",
    format="wav"
)
```

### `ImageArtifact.format` is now required

`ImageArtifact.format` is now a required parameter. Update any code that does not provide a `format` parameter.

#### Before

```python
image_artifact = ImageArtifact(
    b"image_data"
)
```

#### After

```python
image_artifact = ImageArtifact(
    b"image_data",
    format="jpeg"
)
```

### Removed `CsvRowArtifact`

`CsvRowArtifact` has been removed. Use `TextArtifact` instead.

#### Before

```python
artifact = CsvRowArtifact({"name": "John", "age": 30})
print(artifact.value) # {"name": "John", "age": 30}
print(type(artifact.value)) # <class 'dict'>
```

#### After

```python
artifact = TextArtifact("name: John\nage: 30")
print(artifact.value) # name: John\nage: 30
print(type(artifact.value)) # <class 'str'>
```

If you require storing a dictionary as an Artifact, you can use `GenericArtifact` instead.

### `CsvLoader`, `DataframeLoader`, and `SqlLoader` return types

`CsvLoader`, `DataframeLoader`, and `SqlLoader` now return a `list[TextArtifact]` instead of `list[CsvRowArtifact]`.

If you require a dictionary, set a custom `formatter_fn` and then parse the text to a dictionary.

#### Before

```python
results = CsvLoader().load(Path("people.csv").read_text())

print(results[0].value) # {"name": "John", "age": 30}
print(type(results[0].value)) # <class 'dict'>
```

#### After

```python
results = CsvLoader().load(Path("people.csv").read_text())

print(results[0].value) # name: John\nAge: 30
print(type(results[0].value)) # <class 'str'>

# Customize formatter_fn
results = CsvLoader(formatter_fn=lambda x: json.dumps(x)).load(Path("people.csv").read_text())
print(results[0].value) # {"name": "John", "age": 30}
print(type(results[0].value)) # <class 'str'>

dict_results = [json.loads(result.value) for result in results]
print(dict_results[0]) # {"name": "John", "age": 30}
print(type(dict_results[0])) # <class 'dict'>
```

### Moved `ImageArtifact.prompt` and `ImageArtifact.model` to `ImageArtifact.meta`

`ImageArtifact.prompt` and `ImageArtifact.model` have been moved to `ImageArtifact.meta`.

#### Before

```python
image_artifact = ImageArtifact(
    b"image_data",
    format="jpeg",
    prompt="Generate an image of a cat",
    model="DALL-E"
)

print(image_artifact.prompt, image_artifact.model) # Generate an image of a cat, DALL-E
```

#### After

```python
image_artifact = ImageArtifact(
    b"image_data",
    format="jpeg",
    meta={"prompt": "Generate an image of a cat", "model": "DALL-E"}
)

print(image_artifact.meta["prompt"], image_artifact.meta["model"]) # Generate an image of a cat, DALL-E
```

Renamed `GriptapeCloudKnowledgeBaseVectorStoreDriver` to `GriptapeCloudVectorStoreDriver`.

#### Before

```python
from griptape.drivers.griptape_cloud_knowledge_base_vector_store_driver import GriptapeCloudKnowledgeBaseVectorStoreDriver

driver = GriptapeCloudKnowledgeBaseVectorStoreDriver(...)
```

#### After

```python
from griptape.drivers.griptape_cloud_vector_store_driver import GriptapeCloudVectorStoreDriver

driver = GriptapeCloudVectorStoreDriver(...)
```

### `OpenAiChatPromptDriver.response_format` is now a `dict` instead of a `str`.

`OpenAiChatPromptDriver.response_format` is now structured as the `openai` SDK accepts it.

#### Before

```python
driver = OpenAiChatPromptDriver(
    response_format="json_object"
)
```

#### After

```python
driver = OpenAiChatPromptDriver(
    response_format={"type": "json_object"}
)
```

## 0.31.X to 0.32.X

### Removed `DataframeLoader`

`DataframeLoader` has been removed. Use `CsvLoader.parse` or build `TextArtifact`s from the dataframe instead.

#### Before

```python
DataframeLoader().load(df)
```

#### After

```python
# Convert the dataframe to csv bytes and parse it
CsvLoader().parse(bytes(df.to_csv(line_terminator='\r\n', index=False), encoding='utf-8'))
# Or build TextArtifacts from the dataframe
[TextArtifact(row) for row in source.to_dict(orient="records")]
```

### `TextLoader`, `PdfLoader`, `ImageLoader`, and `AudioLoader` now take a `str | PathLike` instead of `bytes`.

#### Before

```python
PdfLoader().load(Path("attention.pdf").read_bytes())
PdfLoader().load_collection([Path("attention.pdf").read_bytes(), Path("CoT.pdf").read_bytes()])
```

#### After

```python
PdfLoader().load("attention.pdf")
PdfLoader().load_collection([Path("attention.pdf"), "CoT.pdf"])
```

### Removed `fileutils.load_file` and `fileutils.load_files`

`griptape.utils.file_utils.load_file` and `griptape.utils.file_utils.load_files` have been removed.
You can now pass the file path directly to the Loader.

#### Before

```python
PdfLoader().load(load_file("attention.pdf").read_bytes())
PdfLoader().load_collection(list(load_files(["attention.pdf", "CoT.pdf"]).values()))
```

```python
PdfLoader().load("attention.pdf")
PdfLoader().load_collection(["attention.pdf", "CoT.pdf"])
```

### Loaders no longer chunk data

Loaders no longer chunk the data after loading it. If you need to chunk the data, use a [Chunker](https://docs.griptape.ai/stable/griptape-framework/data/chunkers/) after loading the data.

#### Before

```python
chunks = PdfLoader().load("attention.pdf")
vector_store.upsert_text_artifacts(
    {
        "griptape": chunks,
    }
)
```

#### After

```python
artifact = PdfLoader().load("attention.pdf")
chunks = Chunker().chunk(artifact)
vector_store.upsert_text_artifacts(
    {
        "griptape": chunks,
    }
)
```

### Removed `MediaArtifact`

`MediaArtifact` has been removed. Use `ImageArtifact` or `AudioArtifact` instead.

#### Before

```python
image_media = MediaArtifact(
    b"image_data",
    media_type="image",
    format="jpeg"
)

audio_media = MediaArtifact(
    b"audio_data",
    media_type="audio",
    format="wav"
)
```

#### After

```python
image_artifact = ImageArtifact(
    b"image_data",
    format="jpeg"
)

audio_artifact = AudioArtifact(
    b"audio_data",
    format="wav"
)
```

### `ImageArtifact.format` is now required

`ImageArtifact.format` is now a required parameter. Update any code that does not provide a `format` parameter.

#### Before

```python
image_artifact = ImageArtifact(
    b"image_data"
)
```

#### After

```python
image_artifact = ImageArtifact(
    b"image_data",
    format="jpeg"
)
```

### Removed `CsvRowArtifact`

`CsvRowArtifact` has been removed. Use `TextArtifact` instead.

#### Before

```python
artifact = CsvRowArtifact({"name": "John", "age": 30})
print(artifact.value) # {"name": "John", "age": 30}
print(type(artifact.value)) # <class 'dict'>
```

#### After

```python
artifact = TextArtifact("name: John\nage: 30")
print(artifact.value) # name: John\nage: 30
print(type(artifact.value)) # <class 'str'>
```

If you require storing a dictionary as an Artifact, you can use `GenericArtifact` instead.

### `CsvLoader`, `DataframeLoader`, and `SqlLoader` return types

`CsvLoader`, `DataframeLoader`, and `SqlLoader` now return a `list[TextArtifact]` instead of `list[CsvRowArtifact]`.

If you require a dictionary, set a custom `formatter_fn` and then parse the text to a dictionary.

#### Before

```python
results = CsvLoader().load(Path("people.csv").read_text())

print(results[0].value) # {"name": "John", "age": 30}
print(type(results[0].value)) # <class 'dict'>
```

#### After

```python
results = CsvLoader().load(Path("people.csv").read_text())

print(type(results)) # <class 'griptape.artifacts.ListArtifact'>
print(results[0].value) # name: John\nAge: 30
print(type(results[0].value)) # <class 'str'>

# Customize formatter_fn
results = CsvLoader(formatter_fn=lambda x: json.dumps(x)).load(Path("people.csv").read_text())
print(results[0].value) # {"name": "John", "age": 30}
print(type(results[0].value)) # <class 'str'>

dict_results = [json.loads(result.value) for result in results]
print(dict_results[0]) # {"name": "John", "age": 30}
print(type(dict_results[0])) # <class 'dict'>
```

### Moved `ImageArtifact.prompt` and `ImageArtifact.model` to `ImageArtifact.meta`

`ImageArtifact.prompt` and `ImageArtifact.model` have been moved to `ImageArtifact.meta`.

#### Before

```python
image_artifact = ImageArtifact(
    b"image_data",
    format="jpeg",
    prompt="Generate an image of a cat",
    model="DALL-E"
)

print(image_artifact.prompt, image_artifact.model) # Generate an image of a cat, DALL-E
```

#### After

```python
image_artifact = ImageArtifact(
    b"image_data",
    format="jpeg",
    meta={"prompt": "Generate an image of a cat", "model": "DALL-E"}
)

print(image_artifact.meta["prompt"], image_artifact.meta["model"]) # Generate an image of a cat, DALL-E
```

## 0.30.X to 0.31.X

### Exceptions Over `ErrorArtifact`s

Drivers, Loaders, and Engines now raise exceptions rather than returning `ErrorArtifact`s.
Update any logic that expects `ErrorArtifact` to handle exceptions instead.

#### Before

```python
artifacts = WebLoader().load("https://www.griptape.ai")

if isinstance(artifacts, ErrorArtifact):
    raise Exception(artifacts.value)
```

#### After

```python
try:
    artifacts = WebLoader().load("https://www.griptape.ai")
except Exception as e:
    raise e
```

### LocalConversationMemoryDriver `file_path` renamed to `persist_file`

`LocalConversationMemoryDriver.file_path` has been renamed to `persist_file` and is now `Optional[str]`. If `persist_file` is not passed as a parameter, nothing will be persisted and no errors will be raised. `LocalConversationMemoryDriver` is now the default driver in the global `Defaults` object.

#### Before

```python
local_driver_with_file = LocalConversationMemoryDriver(
    file_path="my_file.json"
)

local_driver = LocalConversationMemoryDriver()

assert local_driver_with_file.file_path == "my_file.json"
assert local_driver.file_path == "griptape_memory.json"
```

#### After

```python
local_driver_with_file = LocalConversationMemoryDriver(
    persist_file="my_file.json"
)

local_driver = LocalConversationMemoryDriver()

assert local_driver_with_file.persist_file == "my_file.json"
assert local_driver.persist_file is None
```

### Changes to BaseConversationMemoryDriver

`BaseConversationMemoryDriver.driver` has been renamed to `conversation_memory_driver`. Method signatures for `.store` and `.load` have been changed.

#### Before

```python
memory_driver = LocalConversationMemoryDriver()

conversation_memory = ConversationMemory(
    driver=memory_driver
)

load_result: BaseConversationMemory = memory_driver.load()

memory_driver.store(conversation_memory)
```

#### After

```python
memory_driver = LocalConversationMemoryDriver()

conversation_memory = ConversationMemory(
    conversation_memory_driver=memory_driver
)

load_result: tuple[list[Run], dict[str, Any]] = memory_driver.load()

memory_driver.store(
    conversation_memory.runs,
    conversation_memory.meta
)
```
