import os
import json
import importlib
import time
from openai import OpenAI, APITimeoutError
from mlx_lm import load, generate
import libem


def call(*args, **kwargs) -> dict:
    if 'model' in kwargs:
        model_type = kwargs['model']
        if model_type in ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]:
            return openai(*args, **kwargs)
        elif model_type == "llama3":
            return local(*args, **kwargs)
        else:
            raise ValueError(f"Invalid model: {model_type}")


""" OpenAI """

os.environ.setdefault(
    "OPENAI_API_KEY",
    libem.LIBEM_CONFIG.get("OPENAI_API_KEY", "")
)


# LLM call with multiple rounds of tool use
def openai(prompt: str | list | dict,
           tools: list[str] = None,
           context: list = None,
           model: str = "gpt-4o",
           temperature: float = 0.0,
           seed: int = None,
           max_model_call: int = 3,
           ) -> dict:
    if not os.environ.get("OPENAI_API_KEY"):
        raise EnvironmentError(f"OPENAI_API_KEY is not set.")

    client = OpenAI()

    # format the prompt to messages
    match prompt:
        case list():
            messages = prompt
        case dict():
            messages = []
            for role, content in prompt.items():
                if content:
                    messages.append({"role": role, "content": content})
        case str():
            messages = [{"role": "user", "content": prompt}]
        case _:
            raise ValueError(f"Invalid prompt type: {type(prompt)}")

    context = context or []
    messages = context + messages

    # trace variables
    num_model_calls = 0
    num_input_tokens, num_output_tokens = 0, 0
    tool_outputs = {}

    """Start call"""

    if not tools:
        try:
            response = client.chat.completions.create(
                messages=messages,
                model=model,
                temperature=temperature,
                seed=seed,
            )
        except APITimeoutError as e:  # catch timeout error
            raise libem.ModelTimedoutException(e)

        num_model_calls += 1
        num_input_tokens += response.usage.total_tokens - response.usage.completion_tokens
        num_output_tokens += response.usage.completion_tokens
        response_message = response.choices[0].message
    else:
        # Load the tool modules
        tools = [importlib.import_module(tool) for tool in tools]

        # Get the functions from the tools
        available_functions = {
            tool.name: tool.func for tool in tools
        }
        # Get the schema from the tools
        tools = [tool.schema for tool in tools]

        # Call the model
        try:
            response = client.chat.completions.create(
                messages=messages,
                tools=tools,
                tool_choice="auto",
                model=model,
                temperature=temperature,
                seed=seed,
            )
        except APITimeoutError as e:  # catch timeout error
            raise libem.ModelTimedoutException(e)

        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls

        num_model_calls += 1
        num_input_tokens += response.usage.total_tokens - response.usage.completion_tokens
        num_output_tokens += response.usage.completion_tokens

        # Call the tools
        while tool_calls:
            messages.append(response_message)

            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_to_call = available_functions[function_name]
                function_args = json.loads(tool_call.function.arguments)

                libem.debug(f"[{function_name}] {function_args}")

                function_response = function_to_call(**function_args)
                tool_outputs[function_name] = function_response

                messages.append(
                    {
                        "role": "tool",
                        "name": function_name,
                        "content": str(function_response),
                        "tool_call_id": tool_call.id,
                    }
                )

                libem.trace.add({
                    'tool': {
                        "id": tool_call.id,
                        'name': function_name,
                        "arguments": function_args,
                        "response": function_response,
                    }
                })
            tool_calls = []

            if num_model_calls < max_model_call:
                # Call the model again with the tool outcomes
                try:
                    response = client.chat.completions.create(
                        messages=messages,
                        tools=tools,
                        tool_choice="auto",
                        model=model,
                        temperature=temperature,
                        seed=seed,
                    )
                except APITimeoutError as e:  # catch timeout error
                    raise libem.ModelTimedoutException(e)

                response_message = response.choices[0].message
                tool_calls = response_message.tool_calls

                num_model_calls += 1
                num_input_tokens += response.usage.total_tokens - response.usage.completion_tokens
                num_output_tokens += response.usage.completion_tokens

            if num_model_calls == max_model_call:
                libem.debug(f"[model] max call reached: {messages}\n{response_message}")

    """End call"""

    messages.append(response_message)

    libem.trace.add({
        "model": {
            "num_model_calls": num_model_calls,
            "num_input_tokens": num_input_tokens,
            "num_output_tokens": num_output_tokens,
        }
    })

    return {
        "output": response_message.content,
        "messages": messages,
        "tool_outputs": tool_outputs,
    }

""" Local Model """
def local(prompt: str | list | dict,
           tools: list[str] = None,
           context: list = None,
           model: str = "null",
           temperature: float = 0.0,
           seed: int = None,
           max_model_call: int = 3,
           ) -> dict:

    # Load the model using MLX for apple silicon device
    if model == "llama3":
        model_path = "mlx-community/Meta-Llama-3-8B-Instruct-4bit"
    else:
        raise ValueError(f"{model} is not supported.")
    start = time.time()
    model_local, tokenizer = load(model_path)
    print(f"Model loaded in {time.time() - start:.2f} seconds.")

    # format the prompt to messages
    match prompt:
        case list():
            messages = prompt
        case dict():
            messages = []
            for role, content in prompt.items():
                if content:
                    messages.append({"role": role, "content": content})
        case str():
            messages = [{"role": "user", "content": prompt}]
        case _:
            raise ValueError(f"Invalid prompt type: {type(prompt)}")

    context = context or []
    input_text = '\n'.join(f"{prompt['role']}: {prompt['content']}" for prompt in messages)
    messages = context + [input_text]

    if tools:
        raise libem.ToolUseUnsupported("Tool use is not supported")
    else:
        response = generate(model_local, tokenizer, prompt=messages[0], max_tokens=10, temp=temperature)
    return {
        "output": response,
        "messages": "messages is not supported",
        "tool_outputs": "Tool output is not supported",
    }