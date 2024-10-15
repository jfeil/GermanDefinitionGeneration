def prompt_1_ref_free(context_example, context_word, prediction, _):
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

Eingabe:
{context_example} Was ist die Definition von {context_word}?

Ausgabe:
{prediction}


Definition der Metrik:

Die Antwortähnlichkeit wird anhand des Grades der semantischen Ähnlichkeit zwischen der bereitgestellten Ausgabe und der bereitgestellten Frage bewertet. Punkte können auf der Grundlage der schrittweisen Ähnlichkeit in Bedeutung und Beschreibung mit den vorgegebenen Zielen vergeben werden, wobei eine höhere Punktzahl eine größere Übereinstimmung zwischen der bereitgestellten Ausgabe und den vorgegebenen Zielen anzeigt.


Benotungsrubrik:

Ähnlichkeit der Antwort: Nachfolgend finden Sie die Details für die verschiedenen Bewertungsstufen:

- Punktzahl 1: Die Ausgabe weist wenig bis keine semantische Ähnlichkeit mit den vorgegebenen Zielen auf.
- Punktzahl 2: Die Ausgabe weist in einigen Aspekten eine teilweise semantische Ähnlichkeit mit den vorgegebenen Zielen auf.
- Punktzahl 3: Die Ausgabe weist eine mäßige semantische Ähnlichkeit mit den vorgegebenen Zielen auf.
- Punktzahl 4: Die Ausgabe stimmt in den meisten Aspekten mit den vorgegebenen Zielen überein und weist eine große semantische Ähnlichkeit auf.
- Punktzahl 5: Die Ausgabe stimmt in allen wesentlichen Aspekten mit den vorgegebenen Zielen überein.


Beispiele:

Beispiel Eingabe:
Die Liebe überwindet alle Grenzen. Was ist die Definition von Liebe?

Beispiel Ausgabe:
tiefe Zuneigung und Gefühle für jemanden.


Beispiel Punktzahl: 4
Beispiel Begründung: Der bereitgestellte Output ist eng mit dem Ziel verbunden. Sie deckt verschiedene Schlüsselaspekte ab, die in der Zielvorgabe erwähnt werden, einschließlich der Verbundenheit zu einer Person. Auch wenn sie nicht jedes einzelne Detail der Zielvorgabe enthält, weist sie eine erhebliche semantische Ähnlichkeit auf.



Beispiel Eingabe:
Natürlich sind diese Stifte stabil und robust genug, um den täglichen Rettungseinsatz absolvieren zu können. Was ist die Definition von Stifte?

Beispiel Ausgabe:
Schreiblegiergut, meist mit einem Tinten- oder Bleistiftminen gefüllt, zum Schreiben geeignet.


Beispiel Punktzahl: 1
Beispiel Begründung: Die bereitgestellte Ausgabe ist überhaupt nicht mit dem Ziel verbunden. Sie stellt einen vollständig anderen Kontext dar und stimmt nicht mit dem vorgebenen Kontext überein. Diese Ausgabe ist unzureichend.


Fügen Sie keine weiteren neuen Zeilen hinzu. Fügen Sie keine weiteren Felder hinzu.
"""

def prompt_2_ref_free(context_example, context_word, prediction, _):
    """
    same as 1
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


Definition der Metrik:

Die Antwortähnlichkeit wird anhand des Grades der semantischen Ähnlichkeit zwischen der bereitgestellten Ausgabe und der bereitgestellten Frage bewertet. Punkte können auf der Grundlage der schrittweisen Ähnlichkeit in Bedeutung und Beschreibung mit den vorgegebenen Zielen vergeben werden, wobei eine höhere Punktzahl eine größere Übereinstimmung zwischen der bereitgestellten Ausgabe und den vorgegebenen Zielen anzeigt.


Benotungsrubrik:

Ähnlichkeit der Antwort: Nachfolgend finden Sie die Details für die verschiedenen Bewertungsstufen:

- Punktzahl 1: Die Ausgabe weist wenig bis keine semantische Ähnlichkeit mit den vorgegebenen Zielen auf.
- Punktzahl 2: Die Ausgabe weist in einigen Aspekten eine teilweise semantische Ähnlichkeit mit den vorgegebenen Zielen auf.
- Punktzahl 3: Die Ausgabe weist eine mäßige semantische Ähnlichkeit mit den vorgegebenen Zielen auf.
- Punktzahl 4: Die Ausgabe stimmt in den meisten Aspekten mit den vorgegebenen Zielen überein und weist eine große semantische Ähnlichkeit auf.
- Punktzahl 5: Die Ausgabe stimmt in allen wesentlichen Aspekten mit den vorgegebenen Zielen überein.


Beispiele:

Beispiel Eingabe:
Die Liebe überwindet alle Grenzen. Was ist die Definition von Liebe?

Beispiel Ausgabe:
tiefe Zuneigung und Gefühle für jemanden.


Beispiel Punktzahl: 4
Beispiel Begründung: Der bereitgestellte Output ist eng mit dem Ziel verbunden. Sie deckt verschiedene Schlüsselaspekte ab, die in der Zielvorgabe erwähnt werden, einschließlich der Verbundenheit zu einer Person. Auch wenn sie nicht jedes einzelne Detail der Zielvorgabe enthält, weist sie eine erhebliche semantische Ähnlichkeit auf.



Beispiel Eingabe:
Natürlich sind diese Stifte stabil und robust genug, um den täglichen Rettungseinsatz absolvieren zu können. Was ist die Definition von Stifte?

Beispiel Ausgabe:
Schreiblegiergut, meist mit einem Tinten- oder Bleistiftminen gefüllt, zum Schreiben geeignet.


Beispiel Punktzahl: 1
Beispiel Begründung: Die bereitgestellte Ausgabe ist überhaupt nicht mit dem Ziel verbunden. Sie stellt einen vollständig anderen Kontext dar und stimmt nicht mit dem vorgebenen Kontext überein. Diese Ausgabe ist unzureichend.


Fügen Sie keine weiteren neuen Zeilen hinzu. Fügen Sie keine weiteren Felder hinzu.
"""

def prompt_3_ref_free(context_example, context_word, prediction, _):
    """
    Lösung:
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


Definition der Metrik:

Die Antwortähnlichkeit wird anhand des Grades der semantischen Ähnlichkeit zwischen der bereitgestellten Ausgabe und der bereitgestellten Frage bewertet. Punkte können auf der Grundlage der schrittweisen Ähnlichkeit in Bedeutung und Beschreibung mit den vorgegebenen Zielen vergeben werden, wobei eine höhere Punktzahl eine größere Übereinstimmung zwischen der bereitgestellten Ausgabe und den vorgegebenen Zielen anzeigt.


Benotungsrubrik:

Ähnlichkeit der Antwort: Nachfolgend finden Sie die Details für die verschiedenen Bewertungsstufen:

- Punktzahl 1: Die Ausgabe weist wenig bis keine semantische Ähnlichkeit mit den vorgegebenen Zielen auf.
- Punktzahl 2: Die Ausgabe weist in einigen Aspekten eine teilweise semantische Ähnlichkeit mit den vorgegebenen Zielen auf.
- Punktzahl 3: Die Ausgabe weist eine mäßige semantische Ähnlichkeit mit den vorgegebenen Zielen auf.
- Punktzahl 4: Die Ausgabe stimmt in den meisten Aspekten mit den vorgegebenen Zielen überein und weist eine große semantische Ähnlichkeit auf.
- Punktzahl 5: Die Ausgabe stimmt in allen wesentlichen Aspekten mit den vorgegebenen Zielen überein.


Beispiele:

Beispiel Eingabe:
Die Liebe überwindet alle Grenzen. Was ist die Definition von Liebe?

Beispiel Ausgabe:
tiefe Zuneigung und Gefühle für jemanden.


Beispiel Punktzahl: 4
Beispiel Begründung: Der bereitgestellte Output ist eng mit dem Ziel verbunden. Sie deckt verschiedene Schlüsselaspekte ab, die in der Zielvorgabe erwähnt werden, einschließlich der Verbundenheit zu einer Person. Auch wenn sie nicht jedes einzelne Detail der Zielvorgabe enthält, weist sie eine erhebliche semantische Ähnlichkeit auf.



Beispiel Eingabe:
Natürlich sind diese Stifte stabil und robust genug, um den täglichen Rettungseinsatz absolvieren zu können. Was ist die Definition von Stifte?

Beispiel Ausgabe:
Schreiblegiergut, meist mit einem Tinten- oder Bleistiftminen gefüllt, zum Schreiben geeignet.


Beispiel Punktzahl: 1
Beispiel Begründung: Die bereitgestellte Ausgabe ist überhaupt nicht mit dem Ziel verbunden. Sie stellt einen vollständig anderen Kontext dar und stimmt nicht mit dem vorgebenen Kontext überein. Diese Ausgabe ist unzureichend.


Fügen Sie keine weiteren neuen Zeilen hinzu. Fügen Sie keine weiteren Felder hinzu.

Lösung:
"""

def prompt_4_ref_free(context_example, context_word, prediction, _):
    """
    spacing changed 2
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



Definition der Metrik:


Die Antwortähnlichkeit wird anhand des Grades der semantischen Ähnlichkeit zwischen der bereitgestellten Ausgabe und der bereitgestellten Frage bewertet. Punkte können auf der Grundlage der schrittweisen Ähnlichkeit in Bedeutung und Beschreibung mit den vorgegebenen Zielen vergeben werden, wobei eine höhere Punktzahl eine größere Übereinstimmung zwischen der bereitgestellten Ausgabe und den vorgegebenen Zielen anzeigt.


Benotungsrubrik:

Ähnlichkeit der Antwort: Nachfolgend finden Sie die Details für die verschiedenen Bewertungsstufen:

- Punktzahl 1: Die Ausgabe weist wenig bis keine semantische Ähnlichkeit mit den vorgegebenen Zielen auf.
- Punktzahl 2: Die Ausgabe weist in einigen Aspekten eine teilweise semantische Ähnlichkeit mit den vorgegebenen Zielen auf.
- Punktzahl 3: Die Ausgabe weist eine mäßige semantische Ähnlichkeit mit den vorgegebenen Zielen auf.
- Punktzahl 4: Die Ausgabe stimmt in den meisten Aspekten mit den vorgegebenen Zielen überein und weist eine große semantische Ähnlichkeit auf.
- Punktzahl 5: Die Ausgabe stimmt in allen wesentlichen Aspekten mit den vorgegebenen Zielen überein.



Beispiele:


Beispiel Eingabe:
Die Liebe überwindet alle Grenzen. Was ist die Definition von Liebe?

Beispiel Ausgabe:
tiefe Zuneigung und Gefühle für jemanden.



Beispiel Punktzahl: 4
Beispiel Begründung: Der bereitgestellte Output ist eng mit dem Ziel verbunden. Sie deckt verschiedene Schlüsselaspekte ab, die in der Zielvorgabe erwähnt werden, einschließlich der Verbundenheit zu einer Person. Auch wenn sie nicht jedes einzelne Detail der Zielvorgabe enthält, weist sie eine erhebliche semantische Ähnlichkeit auf.



Beispiel Eingabe:
Natürlich sind diese Stifte stabil und robust genug, um den täglichen Rettungseinsatz absolvieren zu können. Was ist die Definition von Stifte?

Beispiel Ausgabe:
Schreiblegiergut, meist mit einem Tinten- oder Bleistiftminen gefüllt, zum Schreiben geeignet.



Beispiel Punktzahl: 1
Beispiel Begründung: Die bereitgestellte Ausgabe ist überhaupt nicht mit dem Ziel verbunden. Sie stellt einen vollständig anderen Kontext dar und stimmt nicht mit dem vorgebenen Kontext überein. Diese Ausgabe ist unzureichend.



Fügen Sie keine weiteren neuen Zeilen hinzu. Fügen Sie keine weiteren Felder hinzu.
"""

def prompt_5_ref_free(context_example, context_word, prediction, _):
    """
    spacing changed
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

Definition der Metrik:
Die Antwortähnlichkeit wird anhand des Grades der semantischen Ähnlichkeit zwischen der bereitgestellten Ausgabe und der bereitgestellten Frage bewertet. Punkte können auf der Grundlage der schrittweisen Ähnlichkeit in Bedeutung und Beschreibung mit den vorgegebenen Zielen vergeben werden, wobei eine höhere Punktzahl eine größere Übereinstimmung zwischen der bereitgestellten Ausgabe und den vorgegebenen Zielen anzeigt.

Benotungsrubrik:
Ähnlichkeit der Antwort: Nachfolgend finden Sie die Details für die verschiedenen Bewertungsstufen:
- Punktzahl 1: Die Ausgabe weist wenig bis keine semantische Ähnlichkeit mit den vorgegebenen Zielen auf.
- Punktzahl 2: Die Ausgabe weist in einigen Aspekten eine teilweise semantische Ähnlichkeit mit den vorgegebenen Zielen auf.
- Punktzahl 3: Die Ausgabe weist eine mäßige semantische Ähnlichkeit mit den vorgegebenen Zielen auf.
- Punktzahl 4: Die Ausgabe stimmt in den meisten Aspekten mit den vorgegebenen Zielen überein und weist eine große semantische Ähnlichkeit auf.
- Punktzahl 5: Die Ausgabe stimmt in allen wesentlichen Aspekten mit den vorgegebenen Zielen überein.

Beispiele:

Beispiel Eingabe:
Die Liebe überwindet alle Grenzen. Was ist die Definition von Liebe?
Beispiel Ausgabe:
tiefe Zuneigung und Gefühle für jemanden.

Beispiel Punktzahl: 4
Beispiel Begründung: Der bereitgestellte Output ist eng mit dem Ziel verbunden. Sie deckt verschiedene Schlüsselaspekte ab, die in der Zielvorgabe erwähnt werden, einschließlich der Verbundenheit zu einer Person. Auch wenn sie nicht jedes einzelne Detail der Zielvorgabe enthält, weist sie eine erhebliche semantische Ähnlichkeit auf.


Beispiel Eingabe:
Natürlich sind diese Stifte stabil und robust genug, um den täglichen Rettungseinsatz absolvieren zu können. Was ist die Definition von Stifte?
Beispiel Ausgabe:
Schreiblegiergut, meist mit einem Tinten- oder Bleistiftminen gefüllt, zum Schreiben geeignet.

Beispiel Punktzahl: 1
Beispiel Begründung: Die bereitgestellte Ausgabe ist überhaupt nicht mit dem Ziel verbunden. Sie stellt einen vollständig anderen Kontext dar und stimmt nicht mit dem vorgebenen Kontext überein. Diese Ausgabe ist unzureichend.


Fügen Sie keine weiteren neuen Zeilen hinzu. Fügen Sie keine weiteren Felder hinzu.
"""

def prompt_6_ref_free(context_example, context_word, prediction, _):
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