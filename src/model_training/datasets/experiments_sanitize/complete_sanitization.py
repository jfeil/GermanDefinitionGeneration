import logging

import regex as re
from datasets import Dataset
from tqdm.auto import tqdm

from src.model_training.datasets.default import DefaultTrainValSet as TemplateTrain
from src.model_training.datasets.default import DefaultTestSet as TemplateTest


@classmethod
def data_preparation(cls, dataset: Dataset, cache=False) -> Dataset:
    if cls.tokenizer is None:
        logging.warning("Tokenizer not available, using no tokenizer")
        dataset = dataset.map(cls._no_tokenizer_preprocessing, batched=True, load_from_cache_file=cache)
    else:
        dataset = dataset.map(cls._preprocessing, batched=True, load_from_cache_file=cache)

    def contains_special(input_string):
        input_string = input_string.replace("[...]", "!!REPLACEMENTPLACEHOLDER!!")
        regex = re.compile(r'[^\p{L}0-9 :"().,;!?\-/\'&%=$§@°\*€†≈~¥\+#]')
        search_result = regex.search(input_string)
        # input_string.replace("!!REPLACEMENTPLACEHOLDER!!", "[...]")

        # Search the input string for any matches
        if search_result:
            return True
        return False

    def contains_math(input_string):
        regex = re.compile(r'<math>')
        search_result = regex.search(input_string)

        # Search the input string for any matches
        if search_result:
            return True
        return False

    special_idx = []
    no_special_idx = []
    new_no_special_idx = []

    math_idx = []
    no_math_idx = []

    for x, datapoint in enumerate(dataset):
        # if contains_special(datapoint['context_sentence']) or contains_special(
        #         datapoint['context_word']) or contains_special(datapoint['gt']):
        #     special_idx += [x]
        #     # print(f"{i['title']}: {i['context_word']} in {i['context_sentence']}\n{i['gt']}")
        # else:
        #     no_special_idx += [x]
        #     if x in old_special_idx:
        #         new_no_special_idx += [x]
        #
        if contains_math(datapoint['context_sentence']) or contains_math(
                datapoint['context_word']) or contains_math(datapoint['gt']):
            math_idx += [x]
            # print(f"{i['title']}: {i['context_word']} in {i['context_sentence']}\n{i['gt']}")
        else:
            no_math_idx += [x]

    dataset = dataset.select(no_math_idx)
    dataset = dataset.filter(lambda x: x['gt'] != "")

    def analyze_dataset(dataset, threshold=0.75):
        def get_embedding(word):
            # Tokenize the input word
            inputs = bert_tokenizer(word, return_tensors='pt').to('cuda')

            # Get the outputs from BERT
            with torch.no_grad():
                outputs = bert_model(**inputs)

            # The outputs are of shape (batch_size, sequence_length, hidden_size)
            # We need the embeddings of the input token(s)
            embeddings = outputs.last_hidden_state

            # If the word is split into multiple tokens, we take the mean of their embeddings
            # Otherwise, we take the embedding of the single token
            word_embedding = embeddings.mean(dim=1).squeeze()

            return word_embedding

        # Function to calculate cosine similarity
        def cosine_similarity(embedding1, embedding2):
            return torch.nn.functional.cosine_similarity(embedding1.unsqueeze(0), embedding2.unsqueeze(0)).item()

        from HanTa import HanoverTagger as ht
        from transformers import BertModel, AutoTokenizer
        import torch

        bert_model = "dbmdz/bert-base-german-uncased"

        # Load the BERT model and tokenizer
        bert_tokenizer = AutoTokenizer.from_pretrained(bert_model)
        bert_model = BertModel.from_pretrained(bert_model)
        bert_model.to('cuda')

        tagger = ht.HanoverTagger('morphmodel_ger.pgz')

        empty_pairs = []
        bad_pairs = []
        for i, (title, context_word) in tqdm(enumerate(zip(dataset["title"], dataset["context_word"]))):
            if title == context_word:
                continue
            if title == "" or context_word == "":
                empty_pairs.append(i)
                continue
            if (a := str(tagger.analyze(title)[0])) != (b := str(tagger.analyze(context_word)[0])):
                if title != b and a != context_word and a != b:
                    word1_embedding = get_embedding(title)
                    word2_embedding = get_embedding(context_word)

                    # Calculate the cosine similarity between the two words
                    similarity = cosine_similarity(word1_embedding, word2_embedding)
                    if similarity < threshold:
                        bad_pairs.append(i)

        return empty_pairs, bad_pairs

    empty, bad = analyze_dataset(dataset)
    dataset = dataset.filter(lambda example, idx: idx not in bad+empty, with_indices=True)

    return dataset


def global_sanitize(input_text):
    input_text = re.sub(r"<small>\[[0-9]*\]<\/small>", "", input_text)
    input_text = re.sub(r"<sup>.*?<\/sup>", "", input_text)
    input_text = re.sub(r"\(\[http:.*?''Internet-Beleg''\)", "", input_text)
    input_text = re.sub(r"<!?--.*?-->", "", input_text)
    input_text = re.sub(r"<\/?span.*?>", "", input_text)
    input_text = re.sub(r"\u200b", "", input_text)
    input_text = re.sub(r"(\u200b|\u200e|\u2060|\u200d|\u200c)", "", input_text)

    def remove_tags(input_text, tag):
        return input_text.replace(f"<{tag}>", "").replace(f"</{tag}>", "")

    input_text = remove_tags(input_text, "small")
    input_text = remove_tags(input_text, "big")
    input_text = remove_tags(input_text, "sub")
    input_text = remove_tags(input_text, "code")
    input_text = remove_tags(input_text, "u")
    input_text = remove_tags(input_text, "s")
    input_text = remove_tags(input_text, "small cap")

    regex = re.compile(r'<(ref|REF).*?</(ref|REF)>')
    input_text = regex.sub('', input_text)
    regex = re.compile(r'<(ref|REF).*?(ref|REF)>')
    input_text = regex.sub('', input_text)
    regex = re.compile(r'<(ref|REF).*?/>')
    input_text = regex.sub('', input_text)
    regex = re.compile(r'<div.*?</div>')
    input_text = regex.sub('', input_text)

    input_text = re.sub(r'\[\[([^\|\]]+)\]\]', r'\1', input_text)

    replacements_k = {
        "ugs.": "umgangssprachlich",
        "refl.": "",  # "reflexiv",
        "trans.": "",  # "transitiv",
        "abw.": "abwertend",
        "übertr.": "übertragen",
        "va.": "vor allem",
        "sal.": "salopp",
        "allg.": "allgemein",
        "auch": "",
        "kein Plural": ""
    }

    replacements = {
        '_': "_",
        '‐': "-",
        '‑': "-",
        '·': "x",
        '̩': "",
        '‛': "'",
        '”': "\"",
        '‟': "\"",
        '(¨)': "",
        'abw.': "abwertend",
        '{{Beispiele fehlen}}': "",
        'CH&LI': "",
        'CURRENTYEAR': "",
        '{{es.}}': "",
        '{{f}}': "",
        '{{fachspr.}}': "fachsprachlich",
        'fam.': "familiär;",
        '{{gM}}': "",
        '{{gM|r}}': "",
        '{{Herkunft}}': "",
        'hist.': "historisch",
        'intrans.': "",
        '{{IPA}}': "",
        'kPl.': "",
        'kSt.': "",
        'geh.': "gehoben",
        'landsch.': "landschaftlich",
        '{{m}}': "m",
        '{{Mask.}}': "",
        '{{md.}}': "",
        '{{n}}': "n",
        'NNBSP': " ",
        'nordd.': "norddeutsch",
        'österr.': "östereichisch:",
        'Pl.': "",
        'QS Bedeutungen': "",
        'QS Herkunft': "",
        'refl.': "",
        'reg.': "",
        'scherzh.': "scherzhaft",
        'schweiz.': "schweizerisch",
        'südd.': "süddeutsch,",
        'trans.': "",
        'übertr.': "übertragen:",
        'ugs.': "umgangssprachlich,",
        'va.': "vor allem.",
        '{{vergleiche}}': "vergleiche",
        'vul.': "vulgär:",
        '′': "'",
        '″': "\"",
        '`': "'",
        '©': "",
        '±': "+-",
        '×': "x",
        '<!--erweitern-->': "",
        '−': "-",
        '⁻': "-",
        '⁄': "/",
        '¹': "",
        '½': "1/2",
        '⅓': "1/3",
        '²': "2",
        '₂': "2",
        '³': "3",
        '₃': "3",
        '⁴': "4",
        '₅': "5",
        '⁶': "6",
        "–": "-",
        "—": "-",
        "…": "...",
        "''": "",
        "’": "'",
        "‘": "\"",
        "‚": "\"",
        "‹": "\"",
        "›": "\"",
        "“": "\"",
        "„": "\"",
        "«": "\"",
        "»": "\"",
        "{{(R)}}": "",
        "fig.": "figurativ:",
        "{{Neutr.}}": "",
        "{{trans.|:}}": "",
        "\x80": "",
        "\x84": "",
        "\x93": "",
        "\x96": "",
        "&#8239;": " ",
        "&#91;sic&#93;": "[sic]",
        "&#x202f;": " ",
        "´": "\"",
        "®": "",
        "→": "",
        "<!--": "",
        "</br>": " ",
        "<br />": " ",
        "<br/>": " ",
        "<br>": " ",
        "<nowiki>[</nowiki>": "",
        "<nowiki>]</nowiki>": "",
        "|sonst|": "|",
        "□": "",
        "vatd.": "veraltend",
        "<nowiki/>": ""
    }

    def replace_k(k_text):
        k_text = k_text.replace("{{K|", "")
        k_text = k_text.replace("}}", "")
        k_text = k_text.split("|")
        use_elements = []
        for el in k_text:
            el = el.strip()
            if el in replacements_k:
                el = replacements_k[el]
            if "=" in el or "" == el:
                continue
            use_elements.append(el)
        if len(use_elements) > 0:
            return ", ".join(use_elements) + ":"
        else:
            return ""

    for key, value in replacements.items():
        input_text = input_text.replace(key, value)

    input_text = remove_tags(input_text, "nowiki")

    def keyword_last(input_text, keyword):
        return re.sub(r"{{" + keyword + "\|.*?}}",
                      lambda u: u.group(0).replace("{{" + keyword + "|", "").replace("}}", "").split("|")[-1],
                      input_text).strip()

    def keyword_delete(input_text, keyword):
        return re.sub(r"{{" + keyword + ".*?}}", "", input_text).strip()

    input_text = re.sub(r"\[\[Datei:.*?]]", "", input_text)

    # filter [[]] and [[|]]
    input_text = re.sub(r'\[\[([^\|\]]+)\|([^\|\]]+)\]\]', lambda u: u.group(2), input_text)

    # filter {{K|}}
    input_text = re.sub(r"{{K\|.*?}}", lambda u: replace_k(u.group(0)), input_text).strip()

    # filter []
    input_text = input_text.replace("[...]", "!!REPLACEMENTPLACEHOLDER!!")
    input_text = re.sub(r'\[([^\|\]]+)\]', r'\1', input_text)
    input_text = input_text.replace("!!REPLACEMENTPLACEHOLDER!!", "[...]")
    # delete completely

    input_text = keyword_last(input_text, "Üt")
    input_text = keyword_last(input_text, "Ü")
    input_text = keyword_last(input_text, "Farbe")
    input_text = keyword_last(input_text, "L")

    input_text = keyword_delete(input_text, "Bibel")
    input_text = keyword_delete(input_text, "Anker")
    input_text = keyword_delete(input_text, "Audio")

    input_text = keyword_delete(input_text, "Internetquelle")
    input_text = keyword_last(input_text, "Polytonisch")
    input_text = keyword_delete(input_text, "QS Bedeutung(en)?")
    input_text = keyword_delete(input_text, "Literatur ")
    input_text = keyword_delete(input_text, "Literatur")

    input_text = keyword_delete(input_text, "Ref-dejure")
    input_text = keyword_delete(input_text, "Wikipedia")
    input_text = keyword_delete(input_text, "Wikisource")
    input_text = keyword_delete(input_text, "W\|")
    input_text = keyword_delete(input_text, "w\|")

    # take last entry only
    input_text = keyword_last(input_text, "Hintergrundfarbe")
    input_text = keyword_last(input_text, "Lautschrift")
    input_text = keyword_last(input_text, "WP")

    input_text = re.sub(r'{{([^|]*?)[,;:]*?\|*?([:,;. ]*?)}}', r'\1\2', input_text)
    input_text = re.sub(r'{{[,;:]*?\|*?.*?}}', '', input_text)
    input_text = re.sub("{{}}", "", input_text)
    input_text = re.sub(".*?}}", "", input_text)
    input_text = input_text.replace(",:", ":")

    return (re.sub("<ref>$", "", input_text.replace("[[", "").replace("]]", "")).
            replace("-->", "").replace("<--", "").replace("<!--", "").
            replace("()", "").replace("ͤ", "").strip(", .:"))


class DefinitionDataset(TemplateTrain):
    train_path = "/home/jfeil/MasterThesis/dataset/v5/train.parquet"

    val_path = "/home/jfeil/MasterThesis/dataset/v5/val.parquet"

    _prepare_data = data_preparation

    @classmethod
    def _sanitize_context(cls, input_text: str) -> str:
        input_text = cls._sanitize_spaces(input_text)
        return global_sanitize(input_text)

    @classmethod
    def _sanitize_word(cls, input_text: str) -> str:
        input_text = cls._sanitize_spaces(input_text)
        input_text = global_sanitize(input_text)
        return re.sub(r"[^\p{L}0-9 ]", "", input_text)

    @classmethod
    def _sanitize_gt(cls, input_text: str) -> str:
        input_text = cls._sanitize_spaces(input_text)
        return global_sanitize(input_text)


class DefinitionTestSet(TemplateTest):
    test_path = "/home/jfeil/MasterThesis/dataset/v5/test.parquet"
    _prepare_data = data_preparation

    @classmethod
    def _sanitize_context(cls, input_text: str) -> str:
        input_text = cls._sanitize_spaces(input_text)
        return global_sanitize(input_text)

    @classmethod
    def _sanitize_word(cls, input_text: str) -> str:
        input_text = cls._sanitize_spaces(input_text)
        input_text = global_sanitize(input_text)
        return re.sub(r"[^\p{L}0-9 ]", "", input_text)

    @classmethod
    def _sanitize_gt(cls, input_text: str) -> str:
        input_text = cls._sanitize_spaces(input_text)
        return global_sanitize(input_text)

    @classmethod
    def _data_loading(cls, shuffle: bool, seed: int, subset_test: float) -> Dataset:
        # noinspection PyTypeChecker
        dataset_test = Dataset.from_parquet(cls.test_path, split="test")

        if shuffle:
            dataset_test = dataset_test.shuffle(seed=seed).flatten_indices()

        if subset_test > 0:
            dataset_test = cls._subset(dataset_test, subset_test)

        return dataset_test

    @classmethod
    def create_dataset(cls, tokenizer, shuffle: bool, seed: int,
                       subset_test: int | float = -1, cache=True) -> Dataset:
        cls.tokenizer = tokenizer
        dataset_test = cls._data_loading(shuffle, seed, subset_test)

        return cls._prepare_data(dataset_test, cache)
