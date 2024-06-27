from aiohttp import ClientTimeout
import aiohttp
import asyncio
from tqdm.notebook import tqdm
import json
from src.evaluation_utils import parse_response


def generate_prompt(data, prompt_function):
    for context_example, context_word, prediction, ground_truth in data:
        yield prompt_function(context_example, context_word, prediction, ground_truth)


async def async_prompting(index, prompt, session, entrypoint="http://127.0.0.1:8080/generate"):
    try:
        json_body = {"inputs": prompt,
                     "parameters": {"temperature": 0.001, "max_new_tokens": 200, "top_p": 0.99, "details": True,
                                    "decoder_input_details": True}}
        async with session.post(url=entrypoint, json=json_body,
                                headers={'Content-Type': 'application/json'}) as response:
            resp = await response.read()
            return index, parse_response(json.loads(resp.decode())["generated_text"])
    except Exception as e:
        print(e)
        print("Unable to get url {} due to {}.".format(entrypoint, e.__class__))

