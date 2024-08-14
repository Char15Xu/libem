import time
import platform

import libem

from transformers import BertTokenizer, BertModel
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
import re

_model, _tokenizer = None, None

""" BERT """


def call(prompt: str | list | dict,
         tools: list[str] = None,
         context: list = None,
         model: str = "bert-base",
         temperature: float = 0.0,
         seed: int = None,
         max_model_call: int = 3,
         ) -> dict:
    global _model, _tokenizer

    if platform.machine() == "arm64" and platform.system() == "Darwin":
        device = "mps"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    
    if model == "bert-base":
        model_name = "bert-base-uncased"
        if _model is None or _tokenizer is None:
            start = time.time()
            _tokenizer = BertTokenizer.from_pretrained(model_name)
            _model = BertModel.from_pretrained(model_name).to(device)
            libem.debug(f"model loaded in {time.time() - start:.2f} seconds.")
        else:
            libem.debug("model loaded from cache")
    elif model == "roberta":
        model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        if _model is None or _tokenizer is None:
            start = time.time()
            _tokenizer = RobertaTokenizer.from_pretrained(model_name)
            _model = RobertaForSequenceClassification.from_pretrained(model_name).to(device)
            libem.debug(f"model loaded in {time.time() - start:.2f} seconds.")
        else:
            libem.debug("model loaded from cache")
    else:
        raise ValueError(f"Invalid model: {model}")

    model, tokenizer = _model, _tokenizer
    model.eval()

    # Extract entities from prompt
    match prompt:
        case list():
            entities_str = prompt[1]['content']
        case dict():
            for role, content in prompt.items():
                if role == 'user' and content:
                    entities_str = content
        case str():
            entities_str = prompt
        case _:
            raise ValueError(f"Invalid prompt type: {type(prompt)}")
    
    entity1_str, entity2_str = entities_str.split("\n")
    entity1 = entity1_str.split(":")[1].strip()
    entity2 = entity2_str.split(":")[1].strip()

    # Tokenize and encode entities
    encoding1 = tokenizer.encode_plus(
        entity1,
        return_tensors='pt'
    ).to(device)

    encoding2 = tokenizer.encode_plus(
        entity2,
        return_tensors='pt'
    ).to(device)


    with torch.no_grad():
        embedding1 = model(**encoding1).last_hidden_state.mean(dim=1)
        embedding2 = model(**encoding2).last_hidden_state.mean(dim=1)

    similarity = torch.nn.functional.cosine_similarity(embedding1, embedding2).item()

    # Determine entities match using semantic similarity
    response = "Yes" if similarity >= 0.50 else "No"

    return {
        "output": response,
        "messages": response,
        "tool_outputs": "Tool output is not supported",
        "stats": {
            "num_model_calls": 1,
            "num_input_tokens": -1,
            "num_output_tokens": -1,
        }
    }


def reset():
    global _model, _tokenizer
    _model, _tokenizer = None, None