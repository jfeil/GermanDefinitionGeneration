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


system_prompts = [
    [
        "Du bist ein Definitionsgenerator. "
        "Du antwortest in Deutsch. "
        "Du bekommst einen Beispielsatz und ein Wort aus dem Satz. "
        "Du antwortest mit der Definition des Wortes im Kontext des Beispiels. "
        "Du antwortest nur mit der Definition. "
        "Du zeigst den Bezug auf den Kontext. "
        "Du antwortest in maximal 10 Worten. "
    ], [
        "Du bist ein Definitionsgenerator. "
        "Du antwortest in Deutsch. "
        "Du bekommst einen Beispielsatz und ein Wort aus dem Satz. "
        "Du antwortest mit der Definition des Wortes im Kontext des Beispiels. "
        "Du antwortest nur mit der Definition. "
        "Du zeigst den Bezug auf den Kontext. "
        "Du antwortest in maximal 5 Worten. "
    ], [
        "Du bist ein Definitionsgenerator. "
        "Du antwortest in Deutsch. "
        "Du bekommst einen Beispielsatz und ein Wort aus dem Satz. "
        "Du antwortest mit der Definition des Wortes im Kontext des Beispiels. "
        "Du antwortest nur mit der Definition. "
        "Du zeigst den Bezug auf den Kontext. "
    ],
]

example_prompts = [
    [
        (
            "Die Liebe überwindet alle Grenzen",
            "Liebe",
            "inniges Gefühl der Zuneigung für jemanden oder für etwas"
        ),
        (
            "Natürlich sind diese Stifte stabil und robust genug, um den täglichen Rettungseinsatz absolvieren zu "
            "können.",
            "Stifte",
            "Rekrut"
        ),
    ],
    [
        (
            "Die Liebe überwindet alle Grenzen",
            "Liebe",
            "inniges Gefühl der Zuneigung für jemanden oder für etwas"
        ),
    ],
    [

    ]
]

default_experiments = [
    # Limit to 5 words, 2 examples
    Experiment(system_prompt=[
        "Du bist ein Definitionsgenerator. "
        "Du antwortest in Deutsch. "
        "Du bekommst einen Beispielsatz und ein Wort aus dem Satz. "
        "Du antwortest mit der Definition des Wortes im Kontext des Beispiels. "
        "Du antwortest nur mit der Definition. "
        "Du zeigst den Bezug auf den Kontext. "
        "Du antwortest in maximal 5 Worten. "
    ],
        question_prompt="\"%s\": Was ist in diesem Kontext die Definition von %s? ",
        example_prompts=[
            ("Die Liebe überwindet alle Grenzen",
             "Liebe",
             "inniges Gefühl der Zuneigung für jemanden oder für etwas"),
            (
                "Natürlich sind diese Stifte stabil und robust genug, um den täglichen Rettungseinsatz absolvieren zu können.",
                "Stifte",
                "Rekrut"),
        ]),

    # Limit to 5 words, 1 example
    Experiment(system_prompt=[
        "Du bist ein Definitionsgenerator. "
        "Du antwortest in Deutsch. "
        "Du bekommst einen Beispielsatz und ein Wort aus dem Satz. "
        "Du antwortest mit der Definition des Wortes im Kontext des Beispiels. "
        "Du antwortest nur mit der Definition. "
        "Du zeigst den Bezug auf den Kontext. "
        "Du antwortest in maximal 5 Worten. "
    ],
        question_prompt="\"%s\": Was ist in diesem Kontext die Definition von %s? ",
        example_prompts=[
            ("Die Liebe überwindet alle Grenzen",
             "Liebe",
             "inniges Gefühl der Zuneigung für jemanden oder für etwas")
        ]),

    # Limit to 5 words, 0 example
    Experiment(system_prompt=[
        "Du bist ein Definitionsgenerator. "
        "Du antwortest in Deutsch. "
        "Du bekommst einen Beispielsatz und ein Wort aus dem Satz. "
        "Du antwortest mit der Definition des Wortes im Kontext des Beispiels. "
        "Du antwortest nur mit der Definition. "
        "Du zeigst den Bezug auf den Kontext. "
        "Du antwortest in maximal 5 Worten. "
    ],
        question_prompt="\"%s\": Was ist in diesem Kontext die Definition von %s? ",
        example_prompts=[]),

    # Limit to 10 words, 2 examples
    Experiment(system_prompt=[
        "Du bist ein Definitionsgenerator. "
        "Du antwortest in Deutsch. "
        "Du bekommst einen Beispielsatz und ein Wort aus dem Satz. "
        "Du antwortest mit der Definition des Wortes im Kontext des Beispiels. "
        "Du antwortest nur mit der Definition. "
        "Du zeigst den Bezug auf den Kontext. "
        "Du antwortest in maximal 10 Worten. "
    ],
        question_prompt="\"%s\": Was ist in diesem Kontext die Definition von %s? ",
        example_prompts=[
            ("Die Liebe überwindet alle Grenzen",
             "Liebe",
             "inniges Gefühl der Zuneigung für jemanden oder für etwas"),
            (
                "Natürlich sind diese Stifte stabil und robust genug, um den täglichen Rettungseinsatz absolvieren zu können.",
                "Stifte",
                "Rekrut"),
        ]),

    # Limit to 10 words, 1 example
    Experiment(system_prompt=[
        "Du bist ein Definitionsgenerator. "
        "Du antwortest in Deutsch. "
        "Du bekommst einen Beispielsatz und ein Wort aus dem Satz. "
        "Du antwortest mit der Definition des Wortes im Kontext des Beispiels. "
        "Du antwortest nur mit der Definition. "
        "Du zeigst den Bezug auf den Kontext. "
        "Du antwortest in maximal 1ß Worten. "
    ],
        question_prompt="\"%s\": Was ist in diesem Kontext die Definition von %s? ",
        example_prompts=[
            ("Die Liebe überwindet alle Grenzen",
             "Liebe",
             "inniges Gefühl der Zuneigung für jemanden oder für etwas"),
        ]),

    # Limit to 10 words, 0 example
    Experiment(system_prompt=[
        "Du bist ein Definitionsgenerator. "
        "Du antwortest in Deutsch. "
        "Du bekommst einen Beispielsatz und ein Wort aus dem Satz. "
        "Du antwortest mit der Definition des Wortes im Kontext des Beispiels. "
        "Du antwortest nur mit der Definition. "
        "Du zeigst den Bezug auf den Kontext. "
        "Du antwortest in maximal 10 Worten. "
    ],
        question_prompt="\"%s\": Was ist in diesem Kontext die Definition von %s? ",
        example_prompts=[]),

    # No Limit, 2 example
    Experiment(system_prompt=[
        "Du bist ein Definitionsgenerator. "
        "Du antwortest in Deutsch. "
        "Du bekommst einen Beispielsatz und ein Wort aus dem Satz. "
        "Du antwortest mit der Definition des Wortes im Kontext des Beispiels. "
        "Du antwortest nur mit der Definition. "
        "Du zeigst den Bezug auf den Kontext. "
    ],
        question_prompt="\"%s\": Was ist in diesem Kontext die Definition von %s? ",
        example_prompts=[
            ("Die Liebe überwindet alle Grenzen",
             "Liebe",
             "inniges Gefühl der Zuneigung für jemanden oder für etwas"),
            (
                "Natürlich sind diese Stifte stabil und robust genug, um den täglichen Rettungseinsatz absolvieren zu können.",
                "Stifte",
                "Rekrut"),
        ]),

    # No Limit, 1 example
    Experiment(system_prompt=[
        "Du bist ein Definitionsgenerator. "
        "Du antwortest in Deutsch. "
        "Du bekommst einen Beispielsatz und ein Wort aus dem Satz. "
        "Du antwortest mit der Definition des Wortes im Kontext des Beispiels. "
        "Du antwortest nur mit der Definition. "
        "Du zeigst den Bezug auf den Kontext. "
    ],
        question_prompt="\"%s\": Was ist in diesem Kontext die Definition von %s? ",
        example_prompts=[
            ("Die Liebe überwindet alle Grenzen",
             "Liebe",
             "inniges Gefühl der Zuneigung für jemanden oder für etwas"),
        ]),
    # No Limit, 0 example
    Experiment(system_prompt=[
        "Du bist ein Definitionsgenerator. "
        "Du antwortest in Deutsch. "
        "Du bekommst einen Beispielsatz und ein Wort aus dem Satz. "
        "Du antwortest mit der Definition des Wortes im Kontext des Beispiels. "
        "Du antwortest nur mit der Definition. "
        "Du zeigst den Bezug auf den Kontext. "
    ],
        question_prompt="\"%s\": Was ist in diesem Kontext die Definition von %s? ",
        example_prompts=[]),
]


def prompt_pattern(context: str, word: str, pattern="\"%s\": Was ist die Definition von %s?") -> str:
    return pattern % (context, word)
