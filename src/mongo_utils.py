from collections import defaultdict
import re
import pymongo

# Connect to MongoDB
mongodb_uri = "mongodb://localhost:27017/"
database_name = "wikipedia_dump"
collection_name = "articles_2"


client = pymongo.MongoClient(mongodb_uri, username='root', password='example')
db = client[database_name]
collection = db[collection_name]


def get_database_entry(title: str):
    query = {"title": title}

    # Retrieving documents that match the query
    return collection.find(query)[0]


def get_text(title: str):
    return get_database_entry(title)['text']


def extract_sentence(text):
    mat = re.match(":\[[0-9]*\]", text)
    if not mat:
        return None, None
    mat2 = re.findall("[0-9]+", mat.group())
    if not mat2:
        return None, None
    return int(mat2[0]), text[mat.span()[1]+1:]


def extract_content(content):

    bedeutungen = defaultdict(lambda: defaultdict(lambda: []))
    beispiele = defaultdict(lambda: defaultdict(lambda: []))

    index = -1
    searching_bed = False
    searching_exa = False
    cache = ""

    for line in content.split("\n"):
        if str(line).startswith('::'):
            line = line.lstrip("::")
        if line.startswith("== ") or line.startswith("=== "):
            index += 1
        if str(line).startswith('*{{') or str(line).startswith('* {{') or str(line).startswith(':{{') or str(line).startswith('::') or str(line).startswith("*''") or str(line).startswith(":''") or str(line).startswith("<!"):
            continue
        if (searching_bed or searching_exa) and str(line).startswith('{{') and line != "{{Bedeutungen}}" and line != "{{Beispiele}}" or str(line).startswith('===') or str(line).startswith("<p "):
            if searching_bed:
                if cache != "":
                    bedeutungen[index][index_2] += [cache.strip()]
                    cache = ""
                searching_bed = False
            if searching_exa:
                if cache != "":
                    beispiele[index][index_2] += [cache.strip()]
                    cache = ""
                searching_exa = False
            continue
        if searching_bed:
            insert_list = bedeutungen
        if searching_exa:
            insert_list = beispiele
        if searching_bed or searching_exa:
            if re.search('{{Beispiele fehlen[^}]*}}', line):
                continue
            if not line.startswith(":[") and cache == "":
                continue
            if line.startswith(":["):
                if cache != "":
                    insert_list[index][index_2] += [cache.strip()]
                    cache = ""
                index_2, text = extract_sentence(line)
                if not index_2:
                    continue
            else:
                text = "\n"+line
            cache += text
        if line == '{{Bedeutungen}}':
            cache = ""
            searching_exa = False
            searching_bed = True
            continue
        if line == '{{Beispiele}}':
            cache = ""
            searching_bed = False
            searching_exa = True
            continue
    
    max_index = 0
    ret_bed = {}
    ret_exa = {}
    keys = []
    for index_headings in bedeutungen:
        if index_headings not in beispiele:
            continue
        for index_contexts in bedeutungen[index_headings]:
            if index_contexts not in beispiele[index_headings]:
                continue
            keys += [index_contexts + max_index]
            ret_bed[str(index_contexts + max_index)] = bedeutungen[index_headings][index_contexts]
            ret_exa[str(index_contexts + max_index)] = beispiele[index_headings][index_contexts]
        if keys:
            max_index = max(keys)

    return ret_bed, ret_exa


def filter_bedeutung(bed):
    bed = re.sub('{{.*}}', '', bed).replace("[", "").replace("]", "")
    return bed.strip()


def create_set(data):
    key_word = data["title"]
    return_data = []

    assert "beispiele" in data

    for key in data["beispiele"]:
        for example in data["beispiele"][key]:
            if example == '{{Beispiele fehlen}}':
                continue
            in_sentence_keyword = re.findall("''[^']+''", "".join(example))
            if len(in_sentence_keyword) >= 1:
                in_sentence_keyword = in_sentence_keyword[0].strip(",").strip(".").strip(";").replace("''", "")
            else:
                in_sentence_keyword = None
            if in_sentence_keyword and len(in_sentence_keyword.split()) > 1:
                # to disable in sentence words with multiple ones e.g. "zwischen Juni und"
                in_sentence_keyword = None
            text = "".join(example)#.replace("''", "")
            text = re.sub(r'<ref>.*?</ref>', '', text)
            return_data += [(
                key_word, in_sentence_keyword, text, filter_bedeutung(data["bedeutungen"][key][0])
            )]
    return return_data


def get_entries(text, token):
    """
    LEGACY
    """
    if token not in text:
        return []
    search_token = '{{' + token + '}}'

    m = re.search(search_token + '\s*:(.*?)(?=\n{{[^{}]+}}|\Z)', text, re.DOTALL)
    if not m:
        return []
    m = m.group()
    m = m.replace(search_token + '\n', '')
    lines = list(filter(lambda item: item.strip(), m.split("\n")))
    return lines


def extract_entries(entries):
    """
    LEGACY
    """
    curr_id = -1
    bedeutungen = defaultdict(lambda: [])
    for i, line in enumerate(entries):
        if re.search('{{Beispiele fehlen\|.*}}', line):
            continue
        if m := re.search(':\[[0-9]+\]', line):
            curr_id = int(re.search('[0-9]+', m.group()).group())
            entry = line.replace(f':[{curr_id}]', '').strip()
            if entry:
                bedeutungen[str(curr_id)] += [entry]
        elif m := re.search('::— ', line):
            if curr_id != -1:
                bedeutungen[str(curr_id)] += [line.replace('::— ', '').strip()]
    return bedeutungen


def clear_database():
    collection.update_many(
        {},
        {"$unset": {"bedeutungen": "", "beispiele": ""}}
    )
