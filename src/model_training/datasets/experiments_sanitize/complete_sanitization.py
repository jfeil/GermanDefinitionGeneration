import regex as re
from datasets import Dataset

from src.model_training.datasets.default import DefaultTrainValSet, DefaultTestSet


def global_sanitize(input_text):
    input_text = input_text.removesuffix("\n}}")

    input_text = re.sub(r"<small>\[[0-9]*]</small>", "", input_text)
    input_text = re.sub(r"<sup>.*?</sup>", "", input_text)
    input_text = re.sub(r"\(\[http:.*?''Internet-Beleg''\)", "", input_text)
    input_text = re.sub(r"<!?--.*?-->", "", input_text)
    input_text = re.sub(r"</?span.*?>", "", input_text)
    input_text = re.sub(r"\u200b", "", input_text)
    input_text = re.sub(r"([​‎⁠‍‌])", "", input_text)

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

    input_text = re.sub(r'\[\[([^|\]]+)]]', r'\1', input_text)

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
        "{{trans.|:}}": "",
        "{{trans.|,}}": "",
        '{{gM}}': "",
        '{{übertr.|:}}': "übertragen:",
        '{{ugs.|,}}': "umgangssprachlich,",
        "{{ugs.|:}}": "umgangssprachlich:",
        "{{ugs.}}": "umgangssprachlich",
        '{{QS Bedeutungen}}': "",
        '{{scherzh.|:}}': "scherzhaft:",
        '{{scherzh.}}': "scherzhaft:",
        '{{geh.|:}}': "gehoben:",
        '{{intrans.|:}}': "",
        '{{refl.|:}}': "",
        '{{kPl.|:}}': "",
        '{{kPl.}}': "",
        '{{fachspr.}}': "fachsprachlich",
        '{{schweiz.|:}}': "schweizerisch:",
        '{{österr.}}': "östereichisch:",
        '{{veraltend|:}}': "veraltend:",
        '{{refl.|,}}': "",
        '{{trans.}}': "",
        '{{NNBSP}}': " ",
        '{{QS Bedeutungen|unbelegt|spr=de}}': "",
        '{{QS Bedeutungen|unbelegt}}': "",
        '{{intrans.|,}}': "",
        '{{va.}}': "vor allem.",
        '{{intrans.|;}}': "",
        '{{abw.|:}}': "abwertend:",
        '{{abw.}}': "abwertend",
        '{{va.|:}}': "vor allem:",
        '{{trans.|;}}': "",
        '{{intrans.}}': "",
        '{{ugs.|}}': "umgangssprachlich",
        '{{vul.|:}}': "vulgär:",
        '{{QS Bedeutungen|stilistisch und semantisch seltsam}}': "",
        '{{refl.}}': "",
        '{{landsch.|:}}': "landschaftlich:",
        '{{geh.|,}}': "gehoben,",
        '{{QS Bedeutungen|keine Synonymauflistung, siehe Hilfe:Bedeutungen}}': "",
        '{{QS Bedeutungen|fehlend}}': "",
        '{{schweiz.|,}}': "schweizerisch,",
        '{{österr.|,}}': "östereichisch,",
        '{{österr.|:}}': "östereichisch:",
        '{{österr.|}}': "östereichisch",
        '{{QS Herkunft|fehlt}}': "",
        '₂': "2",
        '¹': "",
        '<!--erweitern-->': "",
        '²': "2",
        '³': "3",
        '⁴': "4",
        '×': "x",
        '−': "-",
        '_': "_",
        '′': "'",
        '½': "1/2",
        '‐': "-",
        '‑': "-",
        '̩': "",
        '±': "+-",
        '₃': "3",
        '″': "\"",
        '”': "\"",
        '`': "'",
        '‟': "\"",
        '⁻': "-",
        '₅': "5",
        '⁄': "/",
        '⅓': "1/3",
        '(¨)': "",
        '‛': "'",
        '©': "",
        '⁶': "6",
        '·': "x",
        "„": "\"",
        "“": "\"",
        "»": "\"",
        "«": "\"",
        "›": "\"",
        "‹": "\"",
        "‚": "\"",
        "‘": "\"",
        "´": "\"",
        "’": "'",
        "…": "...",
        "–": "-",
        "—": "-",
        "®": "",
        "''": "",
        "<br>": " ",
        "</br>": " ",
        "<br/>": " ",
        "<br />": " ",
        "{{(R)}}": "",
        "&#91;sic&#93;": "[sic]",
        "&#8239;": " ",
        "&#x202f;": " ",
        "<nowiki>[</nowiki>": "",
        "<nowiki>]</nowiki>": "",
        "<!--": "",
        "□": "",
        "{{Neutr.}}": "",
        "\x93": "",
        "\x96": "",
        "\x84": "",
        "\x80": "",
        '{{Beispiele fehlen}}': "",
        '{{CH&LI}}': "",
        '{{CURRENTYEAR}}': "",
        '{{Herkunft}}': "",
        '{{IPA}}': "",
        '{{Mask.}}': "",
        '{{Pl.}}': "",
        '{{es.}}': "",
        '{{fam.|:}}': "familiär:",
        '{{fam.|;}}': "familiär;",
        '{{gM|r}}': "",
        '{{hist.|:}}': "historisch",
        '{{kPl.|,}}': "",
        '{{kSt.}}': "",
        '{{landsch.|}}': "landschaftlich",
        '{{landsch.}}': "landschaftlich",
        '{{m}}': "m",
        '{{nordd.}}': "norddeutsch",
        '{{n}}': "n",
        '{{reg.}}': "",
        '{{scherzh.|,}}': "scherzhaft,",
        '{{schweiz.}}': "schweizerisch",
        '{{südd.|:}}': "süddeutsch:",
        '{{südd.|,}}': "süddeutsch,",
        '{{ugs.|;}}': "umgangssprachlich;",
        '{{va.|,}}': "vor allem,",
        '{{veraltet|:}}': "veraltet:",
        '{{vergleiche}}': "vergleiche",
        '{{übertr.:}}': "übertragen:",
        '{{übertr.}}': "übertragen",
        '{{übertr.|;}}': "übertragen;",
        "{{fig.|:}}": "figurativ:",
        '{{md.}}': "",
        '{{f}}': "",
        "→": "",
        "vatd.": "veraltend",
        "landsch.": "landschaftlich",
        "|sonst|": "|",
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
    input_text = re.sub(r'\[\[([^|\]]+)\|([^|\]]+)]]', lambda u: u.group(2), input_text)

    # filter {{K|}}
    input_text = re.sub(r"{{K\|.*?}}", lambda u: replace_k(u.group(0)), input_text).strip()

    # filter []
    input_text = input_text.replace("[...]", "!!REPLACEMENTPLACEHOLDER!!")
    input_text = re.sub(r'\[([^|\]]+)]', r'\1', input_text)
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

    return (re.sub("<ref>$", "", input_text.replace("[[", "").replace("]]", ""))
            .replace("-->", "").replace("<--", "").replace("<!--", "")
            .replace("()", "").replace("ͤ", ""))


class DefinitionDataset(DefaultTrainValSet):
    train_path = "/home/jfeil/MasterThesis/dataset/v4/train.parquet"

    val_path = "/home/jfeil/MasterThesis/dataset/v4/val.parquet"

    @classmethod
    def _sanitize_context(cls, input_text: str) -> str:
        input_text = cls._sanitize_spaces(input_text)
        return global_sanitize(input_text)

    @classmethod
    def _sanitize_word(cls, input_text: str) -> str:
        input_text = cls._sanitize_spaces(input_text)
        input_text = global_sanitize(input_text)
        return re.sub(r"[^\p{L}0-9]]", "", input_text)

    @classmethod
    def _sanitize_gt(cls, input_text: str) -> str:
        input_text = cls._sanitize_spaces(input_text)
        return global_sanitize(input_text)

    @classmethod
    def _prepare_data(cls, dataset: Dataset, cache=True) -> Dataset:
        if cls.tokenizer is None:
            raise ValueError("Tokenizer must be set before preprocessing.")

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

        return dataset.select(no_math_idx)


class DefinitionTestSet(DefaultTestSet):
    test_path = "/home/jfeil/MasterThesis/dataset/v4/test.parquet"

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
    def _prepare_data(cls, dataset: Dataset, cache=True) -> Dataset:
        if cls.tokenizer is None:
            raise ValueError("Tokenizer must be set before preprocessing.")

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

        return dataset.select(no_math_idx)

    @classmethod
    def create_dataset(cls, tokenizer, shuffle: bool, seed: int,
                       subset_test: int | float = -1, cache=True) -> Dataset:
        cls.tokenizer = tokenizer
        dataset_test = cls._data_loading(shuffle, seed, subset_test)

        return cls._prepare_data(dataset_test, cache)

    @classmethod
    def _preprocessing(cls, examples):
        context_word = [cls._sanitize_word(w) for w in examples['context_word']]
        title = [cls._sanitize_word(w) for w in examples['title']]
        context_sentence = [cls._sanitize_context(w) for w in examples['context_sentence']]
        gt = [cls._sanitize_gt(w) for w in examples['gt']]
        from src.prompting import prompt_pattern
        input_texts = [prompt_pattern(c, w, pattern=cls.prompt_pattern) for c, w in zip(context_sentence, context_word)]
        inputs = cls.tokenizer(input_texts, max_length=512, truncation=True)
        inputs["labels"] = cls.tokenizer(text_target=gt, max_length=128, truncation=True)["input_ids"]
        inputs["prompt"] = input_texts  # cls.tokenizer.batch_decode(inputs["input_ids"])
        inputs["title"] = title
        inputs["context_word"] = context_word
        inputs["context_sentence"] = context_sentence
        inputs["gt"] = gt  # cls.tokenizer.batch_decode(inputs["labels"])

        return inputs

