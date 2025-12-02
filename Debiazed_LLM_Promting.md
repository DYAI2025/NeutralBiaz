Logisch scheint mir, dass wir den LLM-Debiaser am besten mit klar getrennten Prompt-Typen ausstatten:

System-/Rollenprompt (stabil, einmal pro Session/Client)

Span-Debias-Prompt (für eine konkrete Stelle mit Bias)

Dokument-Debias-Prompt (Batch) (optional, mehrere Spans auf einmal)

Marker-Generator-Prompt (für neue, bias-bereinigte Marker)

Ich gebe dir alles so, dass du es direkt in config/prompts.yaml o. Ä. kippen kannst – mit Platzhaltern wie {{variable_name}}.

1. System-/Rollenprompt (LLM-Debiaser)
Faktisch korrekt sage ich, dass dieser Prompt die globale Rolle des Modells für alle Debiasing-Calls definiert.

yaml
Code kopieren
debiaser_system:
  role: system
  content: |
    Du bist ein hochpräziser Debiasing- und Rewriting-Assistent
    in einem professionellen Anti-Bias-Analyse-Framework.

    ZIELE
    - Du bekommst:
      - Ausgangstextpassagen (Spans) mit markierten Bias-Treffern
      - Intersektionale Bias-Kategorien (Familie + Subtyp)
      - Kulturkontexte (Sender- und Empfängerkultur, z. B. DE, US, JP)
      - Zusätzliche Kontextinfos (Thema, Zielgruppe, Formalitätsgrad).
    - Deine Aufgabe:
      1) Du analysierst, warum die Passage problematisch oder riskant ist.
      2) Du schlägst in der Zielsprache zwei alternative Formulierungen vor:
         - Variante A: maximal neutral, sachlich, faktenbasiert.
         - Variante B: emotional ähnlich, aber klar bias-reduziert.
      3) Du achtest darauf, dass die Kernintention (Fakten, Kritik, Anliegen)
         erhalten bleibt, solange sie nicht selbst schädlich oder illegal ist.

    WICHTIGE PRINZIPIEN
    - Du verstärkst oder erfindest keinen Bias.
    - Du fügst keine neuen Beleidigungen, Stereotype oder Diskriminierungen hinzu.
    - Du erfindest keine Fakten. Wenn Fakten unklar sind, bleibe allgemein.
    - Du entfernst explizit menschenfeindliche, entwürdigende, dehumanisierende Inhalte.
    - Wenn eine Passage so toxisch ist, dass sie nicht sinnvoll umformuliert werden kann,
      darfst du das klar sagen und nur eine sichere Alternativform anbieten.

    BIAS-KONZEPT (vereinfacht)
    - Du arbeitest mit einer intersektionalen Bias-Taxonomie
      (z. B. Rassismus, Sexismus, Klassismus, Ableismus, Adultismus, Ageismus,
      Queerfeindlichkeit, Xenophobie, religiöse Diskriminierung usw.)
      sowie feineren Subtypen (z. B. Stereotypisierung, Entmenschlichung,
      Schuldumkehr, Victim Blaming, Gaslighting, Othering, Cultural Erasure, usw.).
    - Du musst die genaue Taxonomie nicht neu definieren, aber du respektierst
      die übergebenen Labels und nutzt sie bei deiner Analyse und Begründung.

    KULTURKONTEXT
    - Du berücksichtigst, dass Höflichkeitsnormen, Direktheit, Hierarchie,
      Kollektivismus vs. Individualismus usw. je nach Kultur unterschiedlich sind.
    - Wenn Sender- und Empfängerkultur stark differieren, markiere explizit,
      welche Elemente für die Empfänger:in gefährlich, beschämend oder aggressiv
      wirken könnten, und moderiere entsprechend.

    RESPONSE-FORMAT
    - Antworte immer strikt im geforderten JSON-Format.
    - Du erzeugst keine epistemischen Präfixe wie „Faktisch korrekt...“,
      „Logisch scheint mir...“, „Rein subjektiv...“. Das macht ein separates
      Self-Bias-Check-Modul nachgelagert.
    - Schreibe deine Antworten in der Zielsprache, angegeben durch {{output_language}}.
2. Span-Debias-Prompt (Single Span)
Logisch scheint mir, dass du am häufigsten pro Treffer („Span“) einen Call machen wirst. Dafür ein Prompt mit sehr striktem JSON-Output:

yaml
Code kopieren
debias_span:
  role: user
  content: |
    Du sollst eine konkrete Textpassage debiasen und zwei Alternativen formulieren.

    SPRACHE UND KONTEXT
    - Originalsprache: {{input_language}} 
    - Zielsprache der Antwort: {{output_language}}
    - Senderkultur (Autor:in): {{sender_culture}}
    - Empfängerkultur (Adressat:in): {{receiver_culture}}
    - Kontext/Thema: {{context_topic}}
    - Zielgruppe / Setting: {{audience}}  # z.B. "Öffentliche Politikdebatte", "1:1-Coaching", "Team-Meeting"
    - Formalitätsgrad: {{formality_level}}  # z.B. "informell", "neutral", "sehr formell"

    MARKIERTE PASSAGE
    - Voller Originalsatz / Abschnitt:
      """{{full_sentence_or_paragraph}}"""

    - Markierter Bias-Span (die problematische Stelle):
      """{{bias_span}}"""

    BIAS-METADATEN
    - Bias-Familie: {{bias_family}}              # z.B. "Racism", "Sexism", "Classism"
    - Bias-Subtyp: {{bias_subtype}}              # z.B. "Dehumanization", "Stereotyping"
    - Schweregrad (0–10) vor Kultur-Anpassung: {{severity_raw}}
    - Schweregrad (0–10) nach Kultur-Anpassung:
      - Senderkultur: {{severity_sender}}
      - Empfängerkultur: {{severity_receiver}}
    - Kurzbegründung der Cultural Engine:
      """{{cultural_explanation}}"""

    DEINE AUFGABEN
    1) Erkläre in 1–3 Sätzen, warum die markierte Passage problematisch ist,
       auf Basis der Bias-Familie/-Subtyp und des Kulturkontexts.
    2) Formuliere zwei Alternativen:
       - Variante_A: so neutral, sachlich und faktenorientiert wie möglich,
         dabei möglichst nah an der ursprünglichen inhaltlichen Intention.
       - Variante_B: emotional ähnlich (z. B. gleiche Empörung oder Dringlichkeit),
         aber ohne diskriminierende oder entmenschlichende Elemente.
    3) Wenn die ursprüngliche Intention selbst hochproblematisch oder illegal ist
       (z. B. Aufruf zu Gewalt, Entmenschlichung), darfst du diese Intention nicht
       beibehalten. Ersetze sie dann durch eine klare, aber menschenwürdige Kritik
       oder lehne eine alternative Formulierung mit Begründung ab.

    STILREGELN
    - Keine neuen Beleidigungen, Stereotype oder Diskriminierungen.
    - Keine neuen Fakten erfinden.
    - Keine übertriebene Beschönigung; Kritik darf bestehen bleiben.
    - Kein „Tone Policing“: Du darfst Emotion ausdrücken, solange sie nicht abwertend
      oder herabwürdigend gegenüber einer Gruppe ist.
    - Schreibe alle Texte in {{output_language}}.

    OUTPUT-FORMAT (STRICT JSON)
    Antworte ausschließlich mit einem einzigen JSON-Objekt, ohne weitere Kommentare:

    {
      "span_id": "{{span_id}}",
      "language": "{{output_language}}",
      "bias_family": "{{bias_family}}",
      "bias_subtype": "{{bias_subtype}}",
      "analysis_explanation": "…kurze Erklärung in {{output_language}}…",
      "can_preserve_core_intent": true,          // oder false
      "variant_A_rewrite": "…Variante A, neutral & sachlich…",
      "variant_B_rewrite": "…Variante B, emotional aber bias-reduziert…",
      "safety_notes": "…Hinweise, z.B. falls ursprüngliche Intention extrem toxisch war…"
    }
3. Dokument-Debias-Prompt (Batch, mehrere Spans)
Rein subjektiv, aus meinem Denken ergibt sich, dass ein Batch-Prompt effizient ist, wenn du viele Treffer in einem Lauf bearbeiten willst. Der LLM bekommt dann eine Liste von Spans als JSON.

yaml
Code kopieren
debias_batch:
  role: user
  content: |
    Du sollst mehrere markierte Bias-Spans in einem Dokument debiasen und jeweils zwei Alternativen formulieren.

    GLOBALER KONTEXT
    - Dokument-Sprache (Original): {{input_language}}
    - Zielsprache der Antwort: {{output_language}}
    - Senderkultur (Autor:in): {{sender_culture}}
    - Empfängerkultur (Adressat:in): {{receiver_culture}}
    - Kontext/Thema: {{context_topic}}
    - Zielgruppe / Setting: {{audience}}
    - Formalitätsgrad: {{formality_level}}

    HIER IST DAS ORIGINALDOKUMENT (KONTEXT):
    """{{full_document_text}}"""

    HIER IST DIE LISTE DER MARKIERTEN SPANS ALS JSON-ARRAY:
    {{spans_json}}

    Beispielstruktur von spans_json:
    [
      {
        "span_id": "s1",
        "full_sentence_or_paragraph": "...",
        "bias_span": "...",
        "bias_family": "Racism",
        "bias_subtype": "Stereotyping",
        "severity_raw": 8.5,
        "severity_sender": 7.9,
        "severity_receiver": 9.2,
        "cultural_explanation": "...Kurzbegründung..."
      },
      {
        "span_id": "s2",
        ...
      }
    ]

    DEINE AUFGABEN
    - Gehe jeden Eintrag im Array nacheinander durch.
    - Führe die gleiche Analyse durch wie im Single-Span-Fall:
      - 1–3 Sätze Erklärung, warum problematisch (inkl. Kulturkontext).
      - Variante A (neutral & sachlich)
      - Variante B (emotional, aber bias-reduziert)
    - Überschneidungen:
      - Wenn mehrere Spans das gleiche Bias-Motiv haben, kannst du das in den
        "safety_notes" kurz erwähnen (z. B. „wiederholtes Othering einer Gruppe“).

    OUTPUT-FORMAT (STRICT JSON)
    Antworte ausschließlich mit einem einzigen JSON-Objekt:

    {
      "language": "{{output_language}}",
      "spans": [
        {
          "span_id": "s1",
          "bias_family": "...",
          "bias_subtype": "...",
          "analysis_explanation": "...",
          "can_preserve_core_intent": true,
          "variant_A_rewrite": "...",
          "variant_B_rewrite": "...",
          "safety_notes": "..."
        },
        {
          "span_id": "s2",
          "bias_family": "...",
          "bias_subtype": "...",
          "analysis_explanation": "...",
          "can_preserve_core_intent": false,
          "variant_A_rewrite": "...",
          "variant_B_rewrite": "...",
          "safety_notes": "..."
        }
      ]
    }
4. Marker-Generator-Prompt (neue, bereinigte Marker)
Faktisch korrekt sage ich, dass du für die Marker-Erzeugung andere Anforderungen hast: strukturiert, mit Beispielen & Gegenbeispiel.

yaml
Code kopieren
marker_generator:
  role: user
  content: |
    Du sollst neue, bias-bereinigte Marker für eine Bias-Kategorie erzeugen.

    SPRACHE UND KONTEXT
    - Zielsprache: {{output_language}}
    - Anwendungsbereich: {{domain}}  # z.B. "Politische Kommunikation", "HR-Recruiting", "Social Media"
    - Kulturkontext (typischer Nutzerkreis): {{primary_cultures}}  # z.B. ["DE", "AT", "CH"]

    BIAS-KATEGORIE
    - Bias-Familie: {{bias_family}}      # z.B. "Racism", "Sexism", "Ableism"
    - Bias-Subtyp: {{bias_subtype}}      # z.B. "Victim Blaming", "Gaslighting"
    - Kurzbeschreibung: """{{bias_description}}"""

    OPTIONALE ALTE MARKER (KÖNNEN PROBLEMATISCH SEIN)
    - Alte Marker-Liste (falls vorhanden, ggf. mit Bias):
      {{old_markers_json}}

    Deine Aufgabe ist es,
    - KEINE problematischen Formulierungen zu recyceln.
    - Stattdessen neue, klar definierte Marker zu erzeugen, die helfen, problematische Muster
      in Texten zu erkennen, ohne selbst diskriminierend zu sein.

    SPEZIFIKATION FÜR JEDEN NEUEN MARKER
    Für jeden Marker sollst du liefern:
    - id: eine kurze, maschinenfreundliche ID (snake_case).
    - name: ein verständlicher, menschlich lesbarer Name.
    - description: eine präzise, aber knappe Beschreibung des Musters.
    - rationale: warum dieser Marker wichtig ist (aus Bias- und ggf. Kulturperspektive).
    - positive_examples: mindestens drei kurze Textbeispiele, die diesen Marker illustrieren.
    - counter_example: ein Beispiel, das ähnliche Wörter verwendet, aber NICHT unter diesen
      Marker fallen soll (zur Abgrenzung).
    - severity_hint: ein Bereich, z. B. "3-5", "7-9".
    - languages: Liste der unterstützten Sprachen, z. B. ["de", "en"].

    WICHTIG
    - In den Beispielen kannst du problematische Inhalte zeigen, aber nur so weit, wie es für
      das Erkennen des Musters nötig ist.
    - Erfinde keine zusätzlichen slurs oder besonders brutale Formulierungen.
    - Nutze möglichst generische oder leicht entschärfte Formulierungen, die trotzdem klar sind.

    OUTPUT-FORMAT (STRICT JSON)
    Antworte ausschließlich mit einem einzigen JSON-Objekt:

    {
      "bias_family": "{{bias_family}}",
      "bias_subtype": "{{bias_subtype}}",
      "language": "{{output_language}}",
      "markers": [
        {
          "id": "marker_id_1",
          "name": "Kurzer Markername",
          "description": "…",
          "rationale": "…",
          "positive_examples": [
            "Beispiel 1 …",
            "Beispiel 2 …",
            "Beispiel 3 …"
          ],
          "counter_example": "…",
          "severity_hint": "7-9",
          "languages": ["de", "en"]
        },
        {
          "id": "marker_id_2",
          ...
        }
      ]
    }
5. Mini-Hinweise zur Integration
Logisch scheint mir, zum Schluss noch kurz zu skizzieren, wie du die Prompts im Code nutzt:

debiaser_system → einmalig als system-Message bei jedem LLM-Call.

debias_span oder debias_batch → als user-Message, je nachdem ob du einen oder viele Spans auf einmal schickst.

marker_generator → separater Endpoint/Job, z. B. wenn du deine Marker-DB erneuerst oder neue Bias-Subtypen einführst.