from datasets import Dataset
import os
import asyncio

base_path = "/home/jfeil/MasterThesis/dataset_distillation/base-v5-unshuffled/"
output_path = "/home/jfeil/MasterThesis/dataset_distillation/v1/"

if not os.path.exists(output_path):
    os.makedirs(output_path)


def load_dataset(path):
    dataset = Dataset.from_parquet(path)
    dataset = dataset.rename_column("gt", "wiktionary_gt")
    return dataset


train_set = load_dataset(base_path + "train.parquet")
test_set = load_dataset(base_path + "test.parquet")
val_set = load_dataset(base_path + "val.parquet")


def prompt_pattern(context: str, word: str, pattern="\"%s\": Was ist die Definition von %s?") -> str:
    return pattern % (context, word)


class Experiment:
    def __init__(self, system_prompt=None, question_prompt="\"%s\": Was ist die Definition von %s? ",
                 example_prompts=(
                         ("Die Liebe überwindet alle Grenzen", "Liebe",
                          "inniges Gefühl der Zuneigung für jemanden oder für etwas"))):
        if system_prompt:
            self.system_prompt = system_prompt
        else:
            self.system_prompt = ["Du bist ein Definitionsgenerator. "
                                  "Du antwortest in Deutsch. "
                                  "Du bekommst einen Beispielsatz und ein Wort aus dem Satz. "
                                  "Du antwortest mit der Definition des Wortes im Kontext des Beispiels. "
                                  "Du antwortest nur mit der Definition. "
                                  "Du zeigst den Bezug auf den Kontext. "
                                  "Du antwortest in maximal 5 Worten. "]
        self.question_prompt = question_prompt
        self.example_prompts = example_prompts

    def create_examples_prompt(self):
        ret_val = []
        for example, word, definition in self.example_prompts:
            ret_val += [prompt_pattern(example, word, pattern=self.question_prompt) + definition]
        return ret_val

    def create_examples(self):
        ret_val = []
        for example, word, definition in self.example_prompts:
            ret_val += [{
                "role": "user",
                "content": self.question_prompt % (example, word)
            },
                {
                    "role": "assistant",
                    "content": definition}
            ]
        return ret_val


default_experiment = Experiment(system_prompt=[
    "Du bist ein Definitionsgenerator. "
    "Du antwortest in Deutsch. "
    "Du bekommst einen Beispielsatz und ein Wort aus dem Satz. "
    "Du antwortest mit der Definition des Wortes im Kontext des Beispiels. "
    "Du antwortest nur mit der Definition. "
    "Du zeigst den Bezug auf den Kontext. "
    "Du antwortest in maximal 10 Worten. "
],
    question_prompt="%s: Was ist in diesem Kontext die Definition von %s?",
    example_prompts=[
        ("Die Liebe überwindet alle Grenzen",
         "Liebe",
         "inniges Gefühl der Zuneigung für jemanden oder für etwas"),
    ])
from tqdm.auto import tqdm
import os
import asyncio
from openai import AsyncOpenAI
import json
import datetime

base_url = "http://localhost:8080/v1"

async_client = AsyncOpenAI(
    base_url=base_url,
    api_key="-"
)


def create_messages(data, experiment):
    return [
        {
            "role": "system",
            "content": " ".join(experiment.system_prompt)
        },
        *experiment.create_examples(),
        {
            "role": "user",
            "content": experiment.question_prompt % (data['context_sentence'], data['context_word']),
        }
    ]


async def async_prompt(row, experiment) -> None:
    return (await async_client.chat.completions.create(
        messages=create_messages(row, experiment),
        model="tgi",
        stream=False,
        max_tokens=512,
        frequency_penalty=1,
        logprobs=False,
        seed=42,
        temperature=0.2,

    )).choices[0].message.content


async def prompt_dataset(dataset, split, experiment, batch_size=512, warm_up=32, checkpoint_dir="checkpoint"):
    prefix = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    responses = []
    tasks = []
    for i in tqdm(dataset):
        tasks.append(async_prompt(i, experiment))
    current_batch = await asyncio.gather(*tasks[0:warm_up])
    responses.append(current_batch)
    with open(f"{checkpoint_dir}/{prefix}_0.json", "w") as f:
        print(current_batch)
        json.dump(current_batch, f)

    for i in tqdm(range(warm_up, len(dataset), batch_size)):
        current_batch = await asyncio.gather(*tasks[i:i + batch_size])
        responses.append(current_batch)
        with open(f"{split}_{checkpoint_dir}/{prefix}_{i}.json", "w") as f:
            json.dump(current_batch, f)
    return responses


responses = asyncio.run(prompt_dataset(test_set, "test", default_experiment))
test_set.add_column("gt", responses)
test_set.to_parquet(f"{output_path}test.parquet")

responses = asyncio.run(prompt_dataset(val_set, "val", default_experiment))
val_set.add_column("gt", responses)
val_set.to_parquet(f"{output_path}val.parquet")

responses = asyncio.run(prompt_dataset(train_set, "train", default_experiment))
train_set.add_column("gt", responses)
train_set.to_parquet(f"{output_path}train.parquet")
