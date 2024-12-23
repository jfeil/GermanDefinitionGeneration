def prompt_mlflow(context_example, context_word, prediction, ground_truth):
    return f"""Task:
You must return the following fields in your response in two lines, one below the other:
score: Your numerical score for the model's answer_similarity based on the rubric
justification: Your reasoning about the model's answer_similarity score

You are an impartial judge. You will be given an input that was sent to a machine
learning model, and you will be given an output that the model produced. You
may also be given additional information that was used by the model to generate the output.

Your task is to determine a numerical score called answer_similarity based on the input and output.
A definition of answer_similarity and a grading rubric are provided below.
You must use the grading rubric to determine your score. You must also justify your score.

Examples could be included below for reference. Make sure to use them as references and to
understand them before completing the task.

Input:
{context_example} Was ist die Definition von {context_word}?

Output:
{prediction}

Additional information used by the model:
key: targets
value:
{ground_truth}

Metric definition:
Answer similarity is evaluated on the degree of semantic similarity of the provided output to the provided targets, which is the ground truth. Scores can be assigned based on the gradual similarity in meaning and description to the provided targets, where a higher score indicates greater alignment between the provided output and provided targets.

Grading rubric:
Answer similarity: Below are the details for different scores:
- Score 1: The output has little to no semantic similarity to the provided targets.
- Score 2: The output displays partial semantic similarity to the provided targets on some aspects.
- Score 3: The output has moderate semantic similarity to the provided targets.
- Score 4: The output aligns with the provided targets in most aspects and has substantial semantic similarity.
- Score 5: The output closely aligns with the provided targets in all significant aspects.

Examples:

Example Output:
MLflow is an open-source platform.

Additional information used by the model:
key: targets
value:
MLflow is an open-source platform for managing the end-to-end machine learning (ML) lifecycle. It was developed by Databricks, a company that specializes in big data and machine learning solutions. MLflow is designed to address the challenges that data scientists and machine learning engineers face when developing, training, and deploying machine learning models.

Example score: 2
Example justification: The provided output is partially similar to the target, as it captures the general idea that MLflow is an open-source platform. However, it lacks the comprehensive details and context provided in the target about MLflow's purpose, development, and challenges it addresses. Therefore, it demonstrates partial, but not complete, semantic similarity.
        

Example Output:
MLflow is an open-source platform for managing machine learning workflows, including experiment tracking, model packaging, versioning, and deployment, simplifying the ML lifecycle.

Additional information used by the model:
key: targets
value:
MLflow is an open-source platform for managing the end-to-end machine learning (ML) lifecycle. It was developed by Databricks, a company that specializes in big data and machine learning solutions. MLflow is designed to address the challenges that data scientists and machine learning engineers face when developing, training, and deploying machine learning models.

Example score: 4
Example justification: The provided output aligns closely with the target. It covers various key aspects mentioned in the target, including managing machine learning workflows, experiment tracking, model packaging, versioning, and deployment. While it may not include every single detail from the target, it demonstrates substantial semantic similarity.
        

You must return the following fields in your response in two lines, one below the other:
score: Your numerical score for the model's answer_similarity based on the rubric
justification: Your reasoning about the model's answer_similarity score

Do not add additional new lines. Do not add any other fields."""


def prompt_default(context_example, context_word, prediction, ground_truth):
    """
    MLflow copy without Input

    """
    return f"""Aufgabe:

Sie müssen die folgenden Felder in Ihrer Antwort in zwei Zeilen untereinander angeben:
Punktzahl: Ihre numerische Punktzahl für die Antwort_Ähnlichkeit des Modells basierend auf der Rubrik
Begründung: Ihre Begründung für die Bewertung der Antwort_Ähnlichkeit des Modells

Sie sind ein unparteiischer Richter. Sie erhalten eine Eingabe, die an ein maschinelles Lernmodell gesendet wurde, und Sie erhalten eine Ausgabe, die das Modell erzeugt hat. Möglicherweise erhalten Sie auch zusätzliche Informationen, die vom Modell verwendet wurden, um die Ausgabe zu erzeugen.
Ihre Aufgabe besteht darin, auf der Grundlage der Eingabe und der Ausgabe einen numerischen Wert namens answer_similarity zu bestimmen. Im Folgenden finden Sie eine Definition von answer_similarity und eine Bewertungsrubrik. Sie müssen die Bewertungsrubrik verwenden, um Ihre Punktzahl zu ermitteln. Außerdem müssen Sie Ihre Punktzahl begründen.
Zur Veranschaulichung können unten Beispiele angeführt werden. Stellen Sie sicher, dass Sie diese als Referenz verwenden und sie verstehen, bevor Sie die Aufgabe lösen.

Ausgabe:
{prediction}

Zusätzliche Informationen, die vom Modell verwendet werden:
key: targets
value:

{ground_truth}


Definition der Metrik:

Die Antwortähnlichkeit wird anhand des Grades der semantischen Ähnlichkeit zwischen der bereitgestellten Ausgabe und den bereitgestellten Zielen, d. h. der Grundwahrheit, bewertet. Punkte können auf der Grundlage der schrittweisen Ähnlichkeit in Bedeutung und Beschreibung mit den vorgegebenen Zielen vergeben werden, wobei eine höhere Punktzahl eine größere Übereinstimmung zwischen der bereitgestellten Ausgabe und den vorgegebenen Zielen anzeigt.


Benotungsrubrik:

Ähnlichkeit der Antwort: Nachfolgend finden Sie die Details für die verschiedenen Bewertungsstufen:

- Punktzahl 1: Die Ausgabe weist wenig bis keine semantische Ähnlichkeit mit den vorgegebenen Zielen auf.
- Punktzahl 2: Die Ausgabe weist in einigen Aspekten eine teilweise semantische Ähnlichkeit mit den vorgegebenen Zielen auf.
- Punktzahl 3: Die Ausgabe weist eine mäßige semantische Ähnlichkeit mit den vorgegebenen Zielen auf.
- Punktzahl 4: Die Ausgabe stimmt in den meisten Aspekten mit den vorgegebenen Zielen überein und weist eine große semantische Ähnlichkeit auf.
- Punktzahl 5: Die Ausgabe stimmt in allen wesentlichen Aspekten mit den vorgegebenen Zielen überein.


Beispiele:

Beispiel Output:
MLflow ist eine Open-Source-Plattform.

Zusätzliche Informationen, die vom Modell verwendet werden:
key: targets
value:
MLflow ist eine Open-Source-Plattform für die Verwaltung des gesamten Lebenszyklus von maschinellem Lernen (ML). Sie wurde von Databricks entwickelt, einem Unternehmen, das sich auf Lösungen für Big Data und maschinelles Lernen spezialisiert hat. MLflow wurde entwickelt, um die Herausforderungen zu bewältigen, denen sich Datenwissenschaftler und Ingenieure für maschinelles Lernen bei der Entwicklung, dem Training und dem Einsatz von Modellen für maschinelles Lernen gegenübersehen.


Beispiel Punktzahl: 2
Beispiel Begründung: Die bereitgestellte Ausgabe ist dem Ziel teilweise ähnlich, da sie die allgemeine Idee, dass MLflow eine Open-Source-Plattform ist, erfasst. Es fehlen jedoch die umfassenden Details und der Kontext, die in der Zielvorgabe über den Zweck von MLflow, seine Entwicklung und die Herausforderungen, die es angeht, enthalten sind. Daher weist sie zwar eine teilweise, aber keine vollständige semantische Ähnlichkeit auf.


Beispiel Ausgabe:
MLflow ist eine Open-Source-Plattform für die Verwaltung von Arbeitsabläufen des maschinellen Lernens, einschließlich der Verfolgung von Experimenten, der Paketierung von Modellen, der Versionierung und der Bereitstellung, wodurch der Lebenszyklus von ML vereinfacht wird.

Zusätzliche Informationen, die vom Modell verwendet werden:
key: targets
value:
MLflow ist eine Open-Source-Plattform für die Verwaltung des gesamten Lebenszyklus von maschinellem Lernen (ML). Sie wurde von Databricks entwickelt, einem Unternehmen, das sich auf Lösungen für Big Data und maschinelles Lernen spezialisiert hat. MLflow wurde entwickelt, um die Herausforderungen zu bewältigen, mit denen Datenwissenschaftler und Ingenieure für maschinelles Lernen bei der Entwicklung, dem Training und dem Einsatz von Modellen für maschinelles Lernen konfrontiert sind.

Beispiel Punktzahl: 4
Beispiel Begründung: Der bereitgestellte Output ist eng mit dem Ziel verbunden. Sie deckt verschiedene Schlüsselaspekte ab, die in der Zielvorgabe erwähnt werden, einschließlich der Verwaltung von Workflows für maschinelles Lernen, der Verfolgung von Experimenten, der Modellpaketierung, der Versionierung und der Bereitstellung. Auch wenn sie nicht jedes einzelne Detail der Zielvorgabe enthält, weist sie eine erhebliche semantische Ähnlichkeit auf.


Sie müssen die folgenden Felder in Ihrer Antwort in zwei Zeilen angeben, eine unter der anderen:
Punktzahl: Ihre numerische Punktzahl für die Antwortähnlichkeit des Modells auf der Grundlage der Rubrik
Begründung: Ihre Begründung für die Bewertung der Antwort_Ähnlichkeit des Modells

Fügen Sie keine weiteren neuen Zeilen hinzu. Fügen Sie keine weiteren Felder hinzu.
"""


def prompt_1(context_example, context_word, prediction, ground_truth):
    """
    custom examples, no input
    """
    return f"""Aufgabe:
Sie müssen die folgenden Felder in Ihrer Antwort in zwei Zeilen untereinander angeben:
Punktzahl: Ihre numerische Punktzahl für die Antwort_Ähnlichkeit des Modells basierend auf der Rubrik
Begründung: Ihre Begründung für die Bewertung der Antwort_Ähnlichkeit des Modells

Sie sind ein unparteiischer Richter. Sie erhalten eine Eingabe, die an ein maschinelles Lernmodell gesendet wurde, und Sie erhalten eine Ausgabe, die das Modell erzeugt hat. Möglicherweise erhalten Sie auch zusätzliche Informationen, die vom Modell verwendet wurden, um die Ausgabe zu erzeugen.
Ihre Aufgabe besteht darin, auf der Grundlage der Eingabe und der Ausgabe einen numerischen Wert namens answer_similarity zu bestimmen. Im Folgenden finden Sie eine Definition von answer_similarity und eine Bewertungsrubrik. Sie müssen die Bewertungsrubrik verwenden, um Ihre Punktzahl zu ermitteln. Außerdem müssen Sie Ihre Punktzahl begründen.
Zur Veranschaulichung können unten Beispiele angeführt werden. Stellen Sie sicher, dass Sie diese als Referenz verwenden und sie verstehen, bevor Sie die Aufgabe lösen.

Ausgabe:
{prediction}

Zusätzliche Informationen, die vom Modell verwendet werden:
key: targets
value:
{ground_truth}


Definition der Metrik:

Die Antwortähnlichkeit wird anhand des Grades der semantischen Ähnlichkeit zwischen der bereitgestellten Ausgabe und den bereitgestellten Zielen, d. h. der Grundwahrheit, bewertet. Punkte können auf der Grundlage der schrittweisen Ähnlichkeit in Bedeutung und Beschreibung mit den vorgegebenen Zielen vergeben werden, wobei eine höhere Punktzahl eine größere Übereinstimmung zwischen der bereitgestellten Ausgabe und den vorgegebenen Zielen anzeigt.


Benotungsrubrik:

Ähnlichkeit der Antwort: Nachfolgend finden Sie die Details für die verschiedenen Bewertungsstufen:

- Punktzahl 1: Die Ausgabe weist wenig bis keine semantische Ähnlichkeit mit den vorgegebenen Zielen auf.
- Punktzahl 2: Die Ausgabe weist in einigen Aspekten eine teilweise semantische Ähnlichkeit mit den vorgegebenen Zielen auf.
- Punktzahl 3: Die Ausgabe weist eine mäßige semantische Ähnlichkeit mit den vorgegebenen Zielen auf.
- Punktzahl 4: Die Ausgabe stimmt in den meisten Aspekten mit den vorgegebenen Zielen überein und weist eine große semantische Ähnlichkeit auf.
- Punktzahl 5: Die Ausgabe stimmt in allen wesentlichen Aspekten mit den vorgegebenen Zielen überein.


Beispiele:

Beispiel Ausgabe:
tiefe Zuneigung und Gefühle für jemanden.

Zusätzliche Informationen, die vom Modell verwendet werden:
key: targets
value:
inniges Gefühl der Zuneigung für jemanden oder für etwas


Beispiel Punktzahl: 4
Beispiel Begründung: Der bereitgestellte Output ist eng mit dem Ziel verbunden. Sie deckt verschiedene Schlüsselaspekte ab, die in der Zielvorgabe erwähnt werden, einschließlich der Verbundenheit zu einer Person. Auch wenn sie nicht jedes einzelne Detail der Zielvorgabe enthält, weist sie eine erhebliche semantische Ähnlichkeit auf.



Beispiel Ausgabe:
Schreiblegiergut, meist mit einem Tinten- oder Bleistiftminen gefüllt, zum Schreiben geeignet.

Zusätzliche Informationen, die vom Modell verwendet werden:
key: targets
value:
Rekrut


Beispiel Punktzahl: 1
Beispiel Begründung: Die bereitgestellte Ausgabe ist überhaupt nicht mit dem Ziel verbunden. Sie stellt einen vollständig anderen Kontext dar und stimmt nicht mit dem vorgebenen Kontext überein. Diese Ausgabe ist unzureichend.


Fügen Sie keine weiteren neuen Zeilen hinzu. Fügen Sie keine weiteren Felder hinzu.
"""


def prompt_2(context_example, context_word, prediction, ground_truth):
    """
    custom examples, structured input
    """
    return f"""Aufgabe:

Sie müssen die folgenden Felder in Ihrer Antwort in zwei Zeilen untereinander angeben:
Punktzahl: Ihre numerische Punktzahl für die Antwort_Ähnlichkeit des Modells basierend auf der Rubrik
Begründung: Ihre Begründung für die Bewertung der Antwort_Ähnlichkeit des Modells

Sie sind ein unparteiischer Richter. Sie erhalten eine Eingabe, die an ein maschinelles Lernmodell gesendet wurde, und Sie erhalten eine Ausgabe, die das Modell erzeugt hat. Möglicherweise erhalten Sie auch zusätzliche Informationen, die vom Modell verwendet wurden, um die Ausgabe zu erzeugen.
Ihre Aufgabe besteht darin, auf der Grundlage der Eingabe und der Ausgabe einen numerischen Wert namens answer_similarity zu bestimmen. Im Folgenden finden Sie eine Definition von answer_similarity und eine Bewertungsrubrik. Sie müssen die Bewertungsrubrik verwenden, um Ihre Punktzahl zu ermitteln. Außerdem müssen Sie Ihre Punktzahl begründen.
Zur Veranschaulichung können unten Beispiele angeführt werden. Stellen Sie sicher, dass Sie diese als Referenz verwenden und sie verstehen, bevor Sie die Aufgabe lösen.

Ausgabe:
{prediction}

Zusätzliche Informationen, die vom Modell verwendet werden:
key: targets
value:
{ground_truth}
key: context word
value:
{context_word}
key: context example
value:
{context_example}

Definition der Metrik:

Die Antwortähnlichkeit wird anhand des Grades der semantischen Ähnlichkeit zwischen der bereitgestellten Ausgabe und den bereitgestellten Zielen, d. h. der Grundwahrheit, bewertet. Punkte können auf der Grundlage der schrittweisen Ähnlichkeit in Bedeutung und Beschreibung mit den vorgegebenen Zielen vergeben werden, wobei eine höhere Punktzahl eine größere Übereinstimmung zwischen der bereitgestellten Ausgabe und den vorgegebenen Zielen anzeigt.


Benotungsrubrik:

Ähnlichkeit der Antwort: Nachfolgend finden Sie die Details für die verschiedenen Bewertungsstufen:

- Punktzahl 1: Die Ausgabe weist wenig bis keine semantische Ähnlichkeit mit den vorgegebenen Zielen auf.
- Punktzahl 2: Die Ausgabe weist in einigen Aspekten eine teilweise semantische Ähnlichkeit mit den vorgegebenen Zielen auf.
- Punktzahl 3: Die Ausgabe weist eine mäßige semantische Ähnlichkeit mit den vorgegebenen Zielen auf.
- Punktzahl 4: Die Ausgabe stimmt in den meisten Aspekten mit den vorgegebenen Zielen überein und weist eine große semantische Ähnlichkeit auf.
- Punktzahl 5: Die Ausgabe stimmt in allen wesentlichen Aspekten mit den vorgegebenen Zielen überein.


Beispiele:

Beispiel Ausgabe:
tiefe Zuneigung und Gefühle für jemanden.

Zusätzliche Informationen, die vom Modell verwendet werden:
key: targets
value:
inniges Gefühl der Zuneigung für jemanden oder für etwas
key: context word
value:
Liebe
key: context example
value:
Die Liebe überwindet alle Grenzen.


Beispiel Punktzahl: 4
Beispiel Begründung: Der bereitgestellte Output ist eng mit dem Ziel verbunden. Sie deckt verschiedene Schlüsselaspekte ab, die in der Zielvorgabe erwähnt werden, einschließlich der Verbundenheit zu einer Person. Auch wenn sie nicht jedes einzelne Detail der Zielvorgabe enthält, weist sie eine erhebliche semantische Ähnlichkeit auf.



Beispiel Ausgabe:
Schreiblegiergut, meist mit einem Tinten- oder Bleistiftminen gefüllt, zum Schreiben geeignet.

Zusätzliche Informationen, die vom Modell verwendet werden:
key: targets
value:
Rekrut
key: context word
value:
Stifte
key: context example
value:
Natürlich sind diese Stifte stabil und robust genug, um den täglichen Rettungseinsatz absolvieren zu können.


Beispiel Punktzahl: 1
Beispiel Begründung: Die bereitgestellte Ausgabe ist überhaupt nicht mit dem Ziel verbunden. Sie stellt einen vollständig anderen Kontext dar und stimmt nicht mit dem vorgebenen Kontext überein. Diese Ausgabe ist unzureichend.


Fügen Sie keine weiteren neuen Zeilen hinzu. Fügen Sie keine weiteren Felder hinzu.
"""

def prompt_3(context_example, context_word, prediction, ground_truth):
    """
    custom examples, structured input
    """
    return f"""Aufgabe:
Sie müssen die folgenden Felder in Ihrer Antwort in zwei Zeilen untereinander angeben:
Punktzahl: Ihre numerische Punktzahl für die Antwort_Ähnlichkeit des Modells basierend auf der Rubrik
Begründung: Ihre Begründung für die Bewertung der Antwort_Ähnlichkeit des Modells

Sie sind ein unparteiischer Richter. Sie erhalten eine Eingabe, die an ein maschinelles Lernmodell gesendet wurde, und Sie erhalten eine Ausgabe, die das Modell erzeugt hat. Möglicherweise erhalten Sie auch zusätzliche Informationen, die vom Modell verwendet wurden, um die Ausgabe zu erzeugen.
Ihre Aufgabe besteht darin, auf der Grundlage der Eingabe und der Ausgabe einen numerischen Wert namens answer_similarity zu bestimmen. Im Folgenden finden Sie eine Definition von answer_similarity und eine Bewertungsrubrik. Sie müssen die Bewertungsrubrik verwenden, um Ihre Punktzahl zu ermitteln. Außerdem müssen Sie Ihre Punktzahl begründen.
Zur Veranschaulichung können unten Beispiele angeführt werden. Stellen Sie sicher, dass Sie diese als Referenz verwenden und sie verstehen, bevor Sie die Aufgabe lösen.

Eingabe:
{context_example} Was ist die Definition von {context_word}?

Ausgabe:
{prediction}

Zusätzliche Informationen, die vom Modell verwendet werden:
key: targets
value:
{ground_truth}


Definition der Metrik:

Die Antwortähnlichkeit wird anhand des Grades der semantischen Ähnlichkeit zwischen der bereitgestellten Ausgabe und den bereitgestellten Zielen, d. h. der Grundwahrheit, bewertet. Punkte können auf der Grundlage der schrittweisen Ähnlichkeit in Bedeutung und Beschreibung mit den vorgegebenen Zielen vergeben werden, wobei eine höhere Punktzahl eine größere Übereinstimmung zwischen der bereitgestellten Ausgabe und den vorgegebenen Zielen anzeigt.


Benotungsrubrik:

Ähnlichkeit der Antwort: Nachfolgend finden Sie die Details für die verschiedenen Bewertungsstufen:

- Punktzahl 1: Die Ausgabe weist wenig bis keine semantische Ähnlichkeit mit den vorgegebenen Zielen auf.
- Punktzahl 2: Die Ausgabe weist in einigen Aspekten eine teilweise semantische Ähnlichkeit mit den vorgegebenen Zielen auf.
- Punktzahl 3: Die Ausgabe weist eine mäßige semantische Ähnlichkeit mit den vorgegebenen Zielen auf.
- Punktzahl 4: Die Ausgabe stimmt in den meisten Aspekten mit den vorgegebenen Zielen überein und weist eine große semantische Ähnlichkeit auf.
- Punktzahl 5: Die Ausgabe stimmt in allen wesentlichen Aspekten mit den vorgegebenen Zielen überein.


Beispiele:

Beispiel Ausgabe:
tiefe Zuneigung und Gefühle für jemanden.

Zusätzliche Informationen, die vom Modell verwendet werden:
key: targets
value:
inniges Gefühl der Zuneigung für jemanden oder für etwas


Beispiel Punktzahl: 4
Beispiel Begründung: Der bereitgestellte Output ist eng mit dem Ziel verbunden. Sie deckt verschiedene Schlüsselaspekte ab, die in der Zielvorgabe erwähnt werden, einschließlich der Verbundenheit zu einer Person. Auch wenn sie nicht jedes einzelne Detail der Zielvorgabe enthält, weist sie eine erhebliche semantische Ähnlichkeit auf.



Beispiel Ausgabe:
Schreiblegiergut, meist mit einem Tinten- oder Bleistiftminen gefüllt, zum Schreiben geeignet.

Zusätzliche Informationen, die vom Modell verwendet werden:
key: targets
value:
Rekrut


Beispiel Punktzahl: 1
Beispiel Begründung: Die bereitgestellte Ausgabe ist überhaupt nicht mit dem Ziel verbunden. Sie stellt einen vollständig anderen Kontext dar und stimmt nicht mit dem vorgebenen Kontext überein. Diese Ausgabe ist unzureichend.


Fügen Sie keine weiteren neuen Zeilen hinzu. Fügen Sie keine weiteren Felder hinzu.
"""


def prompt_4(context_example, context_word, prediction, ground_truth):
    """
    MLflow prompt translated

    """
    return f"""Aufgabe:

Sie müssen die folgenden Felder in Ihrer Antwort in zwei Zeilen untereinander angeben:
Punktzahl: Ihre numerische Punktzahl für die Antwort_Ähnlichkeit des Modells basierend auf der Rubrik
Begründung: Ihre Begründung für die Bewertung der Antwort_Ähnlichkeit des Modells

Sie sind ein unparteiischer Richter. Sie erhalten eine Eingabe, die an ein maschinelles Lernmodell gesendet wurde, und Sie erhalten eine Ausgabe, die das Modell erzeugt hat. Möglicherweise erhalten Sie auch zusätzliche Informationen, die vom Modell verwendet wurden, um die Ausgabe zu erzeugen.
Ihre Aufgabe besteht darin, auf der Grundlage der Eingabe und der Ausgabe einen numerischen Wert namens answer_similarity zu bestimmen. Im Folgenden finden Sie eine Definition von answer_similarity und eine Bewertungsrubrik. Sie müssen die Bewertungsrubrik verwenden, um Ihre Punktzahl zu ermitteln. Außerdem müssen Sie Ihre Punktzahl begründen.
Zur Veranschaulichung können unten Beispiele angeführt werden. Stellen Sie sicher, dass Sie diese als Referenz verwenden und sie verstehen, bevor Sie die Aufgabe lösen.

Eingabe:
{context_example} Was ist die Definition von {context_word}?

Ausgabe:
{prediction}

Zusätzliche Informationen, die vom Modell verwendet werden:
key: targets
value:
{ground_truth}


Definition der Metrik:

Die Antwortähnlichkeit wird anhand des Grades der semantischen Ähnlichkeit zwischen der bereitgestellten Ausgabe und den bereitgestellten Zielen, d. h. der Grundwahrheit, bewertet. Punkte können auf der Grundlage der schrittweisen Ähnlichkeit in Bedeutung und Beschreibung mit den vorgegebenen Zielen vergeben werden, wobei eine höhere Punktzahl eine größere Übereinstimmung zwischen der bereitgestellten Ausgabe und den vorgegebenen Zielen anzeigt.


Benotungsrubrik:

Ähnlichkeit der Antwort: Nachfolgend finden Sie die Details für die verschiedenen Bewertungsstufen:

- Punktzahl 1: Die Ausgabe weist wenig bis keine semantische Ähnlichkeit mit den vorgegebenen Zielen auf.
- Punktzahl 2: Die Ausgabe weist in einigen Aspekten eine teilweise semantische Ähnlichkeit mit den vorgegebenen Zielen auf.
- Punktzahl 3: Die Ausgabe weist eine mäßige semantische Ähnlichkeit mit den vorgegebenen Zielen auf.
- Punktzahl 4: Die Ausgabe stimmt in den meisten Aspekten mit den vorgegebenen Zielen überein und weist eine große semantische Ähnlichkeit auf.
- Punktzahl 5: Die Ausgabe stimmt in allen wesentlichen Aspekten mit den vorgegebenen Zielen überein.


Beispiele:

Beispiel Output:
MLflow ist eine Open-Source-Plattform.

Zusätzliche Informationen, die vom Modell verwendet werden:
key: targets
value:
MLflow ist eine Open-Source-Plattform für die Verwaltung des gesamten Lebenszyklus von maschinellem Lernen (ML). Sie wurde von Databricks entwickelt, einem Unternehmen, das sich auf Lösungen für Big Data und maschinelles Lernen spezialisiert hat. MLflow wurde entwickelt, um die Herausforderungen zu bewältigen, denen sich Datenwissenschaftler und Ingenieure für maschinelles Lernen bei der Entwicklung, dem Training und dem Einsatz von Modellen für maschinelles Lernen gegenübersehen.


Beispiel Punktzahl: 2
Beispiel Begründung: Die bereitgestellte Ausgabe ist dem Ziel teilweise ähnlich, da sie die allgemeine Idee, dass MLflow eine Open-Source-Plattform ist, erfasst. Es fehlen jedoch die umfassenden Details und der Kontext, die in der Zielvorgabe über den Zweck von MLflow, seine Entwicklung und die Herausforderungen, die es angeht, enthalten sind. Daher weist sie zwar eine teilweise, aber keine vollständige semantische Ähnlichkeit auf.


Beispiel Ausgabe:
MLflow ist eine Open-Source-Plattform für die Verwaltung von Arbeitsabläufen des maschinellen Lernens, einschließlich der Verfolgung von Experimenten, der Paketierung von Modellen, der Versionierung und der Bereitstellung, wodurch der Lebenszyklus von ML vereinfacht wird.

Zusätzliche Informationen, die vom Modell verwendet werden:
key: targets
value:
MLflow ist eine Open-Source-Plattform für die Verwaltung des gesamten Lebenszyklus von maschinellem Lernen (ML). Sie wurde von Databricks entwickelt, einem Unternehmen, das sich auf Lösungen für Big Data und maschinelles Lernen spezialisiert hat. MLflow wurde entwickelt, um die Herausforderungen zu bewältigen, mit denen Datenwissenschaftler und Ingenieure für maschinelles Lernen bei der Entwicklung, dem Training und dem Einsatz von Modellen für maschinelles Lernen konfrontiert sind.

Beispiel Punktzahl: 4
Beispiel Begründung: Der bereitgestellte Output ist eng mit dem Ziel verbunden. Sie deckt verschiedene Schlüsselaspekte ab, die in der Zielvorgabe erwähnt werden, einschließlich der Verwaltung von Workflows für maschinelles Lernen, der Verfolgung von Experimenten, der Modellpaketierung, der Versionierung und der Bereitstellung. Auch wenn sie nicht jedes einzelne Detail der Zielvorgabe enthält, weist sie eine erhebliche semantische Ähnlichkeit auf.


Sie müssen die folgenden Felder in Ihrer Antwort in zwei Zeilen angeben, eine unter der anderen:
Punktzahl: Ihre numerische Punktzahl für die Antwortähnlichkeit des Modells auf der Grundlage der Rubrik
Begründung: Ihre Begründung für die Bewertung der Antwort_Ähnlichkeit des Modells

Fügen Sie keine weiteren neuen Zeilen hinzu. Fügen Sie keine weiteren Felder hinzu.
"""
