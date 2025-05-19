# Local VLLM Agent Code

## Integrating Pydantic AI with VLLM's OpenAI Compatible Server

```python
#####################################################
### SERVER MODEL SETUP
### Local OpenAI-like server with VLLM (when I have some money)

httpx_client = httpx.AsyncClient(verify=False)
openai_like_client = AsyncOpenAI(
    api_key=os.environ.get("API_KEY"),
    base_url=os.environ.get(
        "SERVER_BASE_URL"
    ),
    http_client=httpx_client,
    timeout=600.0,  # set this to a reasonable default
    max_retries=1,
)


async def get_models() -> list[str]:
    """Get the model(s) available.

    Note:
        vllm currently only supports one model per endpoint.
    """
    models = await openai_like_client.models.list()
    output = [item.id for item in models.data if item.object == "model"]
    return output


MODELS_AVAILABLE: Final[list[str]] = asyncio.run(get_models())
if len(MODELS_AVAILABLE) <= 0:
    msg = "ERROR: Server doesn't have any models listed!"
    raise RuntimeError(msg)
print(MODELS_AVAILABLE)

# # # # # # # # # # # # # # # #
# AI MODEL SETUP              #
# MONKEYPATCHING THE REQUEST  #
# # # # # # # # # # # # # # # #

# NOTE(jdwh08): VLLM Documentation @ https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#chat-api
# https://github.com/pydantic/pydantic-ai/blob/613dac881d98eb2848a6abddc9916028f8c8184c/pydantic_ai_slim/pydantic_ai/models/openai.py#L239

@mlflow.trace(...)
async def _completions_create(
    self,
    messages,
    stream,
    model_settings,
    model_request_parameters,
):
    tools = self._get_tools(model_request_parameters)
    print(f"TOOLS: {tools!s}")

    # standalone function to make it easier to override
    if not tools:
        tool_choice: Literal[none, required, auto] | None = None
    elif not model_request_parameters.allow_text_result:
        tool_choice = "required"
    else:
        tool_choice = "auto"

    openai_messages = []
    for m in messages:
        async for msg in self._map_message(m):
            openai_messages.append(msg)

    try:
        response_stream = await self.client.chat.completions.create(
            model=self._model_name,
            messages=openai_messages,
            n=1,
            parallel_tool_calls=model_settings.get("parallel_tool_calls", NOT_GIVEN),
            tools=tools or NOT_GIVEN,
            tool_choice=tool_choice or NOT_GIVEN,
            stream=stream,
            stream_options={"include_usage": True} if stream else NOT_GIVEN,
            max_tokens=model_settings.get(
                "max_tokens", NOT_GIVEN
            ),  # NOTE(jdwh08): Must be monkeypatched from max_completion_tokens
            temperature=model_settings.get("temperature", NOT_GIVEN),
            top_p=model_settings.get("top_p", NOT_GIVEN),
            timeout=model_settings.get("timeout", NOT_GIVEN),
            seed=model_settings.get("seed", NOT_GIVEN),
            presence_penalty=model_settings.get("presence_penalty", NOT_GIVEN),
            frequency_penalty=model_settings.get("frequency_penalty", NOT_GIVEN),
            logit_bias=model_settings.get("logit_bias", NOT_GIVEN),
            reasoning_effort=model_settings.get("openai_reasoning_effort", NOT_GIVEN),
            user=model_settings.get("openai_user", NOT_GIVEN),
        )
    except APIStatusError as e:
        if (status_code := e.status_code) >= 400:
            raise ModelHTTPError(
                status_code=status_code, model_name=self.model_name, body=e.body
            ) from e
        raise
    else:
        return response_stream


# MONKEYPATCH
OpenAIModel._completions_create = _completions_create

openai_like_provider = OpenAIProvider(openai_client=openai_like_client)
model = OpenAIModel(
    MODELS_AVAILABLE[0],
    provider=openai_like_provider,
)

```

### Gemini or other Open-AI compatible endpoints are similar

``` python
# If we ever get Gemini and they fix it to allow async / logging
openai_like_client = AsyncOpenAI(
    api_key=os.environ.get("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)
openai_like_provider = OpenAIProvider(openai_client=openai_like_client)
model = OpenAIModel(
    "gemini-2.0-flash",
    provider=openai_like_provider,
)
```

## TODO

### Structured output with constrained token generation for Local Models

- The PYDANTIC AI people don't have structured output as a priority.
- We'll probably need to do this ourselves.
- Fortunately, vLLM (and HFTransformers I guess) have a field which
accepts BaseModel!
