# Pydantic AI Notes

## Validators

Use a @agentname.result_validator function with context and result to validate
Works best with structured output Pydnatic Model

- Note that structured output for this is awful (tool call -> retry approach)

## Dependencies

Define a dataclass for your dependencies. Specify it to the agent.
If you want dynamic system prompts...

1. Use a ```python @agentname.system_prompt(ctx: RunContext[Deps]) -> str```
2. Pull variables from the ctx in a f-string.
3. Don't use a static system prompt defined in Agent.

## Tools

Takes a dependencies object. If you don't need that, use tool_plain.
All parameter(s) besides the context object must be filled by the model.
Pydnatic will continue to prompt until the model decides it does not need a tool call.

Tool is probably best to be deterministic.

Tool can raise ModelRetry from PydanticAI in cases where the input(s) are not valid
E.g, `python if input not in list: raise ModelRetry(f"Invalid input: {input}")`

## Agent

Agent should be used for intermediate steps which transform input to output.

YTBCodes highly suggests changing the agent's `result tool name` and `result tool description`
so that it clearly outputs what is from the prompt.
E.g., if system prompt output is "identify the user intent"
set the result_tool_name as `user_intent` and result tool description as `!Log! the user's intended action and whether it is valid.`
Agent calls the tool to "log" the response (lol) so that it thinks it is a tool call.
This makes it easier for open source models to use.

YTBCodes also suggests having the task definition in the system prompt
E.g., "the title must be present in this list" -> list from dynamic system prompt
Find the title which is the closest match...

Final executor agent can orchestrate control flow which executes other "agents" based on
conditions.

## Compared to other frameworks

Pydantic focuses heavily on the concept of "how we interact with one agent."
But its control flow being python only, and not a state graph, is worse for complex projects.

- Pydantic has NO OPINION on orchestration, just focuses on run_sync()

## Sources

YTB Codes

- [x] Pydantic AI <https://www.youtube.com/watch?v=ferNt5JTaGQ>
- [ ] TODO: Pydantic Graph <https://www.youtube.com/watch?v=ePp7Gq2bJjE>
- [ ] TODO: Pydantic w/ Langgraph? <https://www.youtube.com/watch?v=P3qH5GVIxD0>
