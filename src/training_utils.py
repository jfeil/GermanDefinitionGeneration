from src.mongo_utils import get_all_entries, create_set


def generator(split="train", limit=0):
    pointer = get_all_entries(split=split, limit=limit)
    buffer = []

    for data in pointer:
        if not buffer:
            buffer += create_set(data, replace_none=True)
        if len(buffer) == 0:
            continue
        title, context_word, context_example, gt = buffer.pop(0)
        yield {'input': [title, context_word, context_example], 'gt': gt}
