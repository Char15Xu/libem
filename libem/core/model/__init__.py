from libem.core.model import (
    openai, llama, bert
)
from libem.core import exec
import libem
import libem

def call(*args, **kwargs) -> dict:
        return exec.run_async_task(
            async_call(*args, **kwargs)
    )


async def async_call(*args, **kwargs) -> dict:
    match kwargs.get("model", ""):
        case "llama3" | "llama3.1":
            return llama.call(*args, **kwargs)
        case "bert-base" | "roberta":
            return bert.call(*args, **kwargs)
        case _:
            return openai.async_call(*args, **kwargs)


def reset():
    openai.reset()
    bert.reset()
    llama.reset()
