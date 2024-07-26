from libem.core.model import (
    openai, llama, claude
)
from libem.core import exec


def call(*args, **kwargs) -> dict:
    return exec.run_async_task(
        async_call(*args, **kwargs)
    )


async def async_call(*args, **kwargs) -> dict:
    if kwargs.get("model", "") == "llama3":
        return llama.call(*args, **kwargs)
    elif kwargs.get("model", "") == "claude-3-5-sonnet-20240620":
        return claude.call(*args, **kwargs)
    else:
        return await openai.async_call(*args, **kwargs)


def reset():
    openai.reset()
    llama.reset()
