"""
Prompt management system with YAML templates and variable substitution.
"""

import logging
import re
from pathlib import Path
from string import Template
from typing import Dict, List, Optional, Any, Union

import yaml

from .models import PromptTemplate
from ..core.exceptions import BiasEngineError


logger = logging.getLogger(__name__)


class PromptError(BiasEngineError):
    """Base exception for prompt-related errors."""
    pass


class TemplateNotFoundError(PromptError):
    """Raised when a template is not found."""
    pass


class VariableError(PromptError):
    """Raised when template variables are invalid."""
    pass


class PromptManager:
    \"\"\"
    Manages prompt templates with variable substitution.

    Supports the 4 core prompt types from the specification:
    - debiaser_system (system role prompt)
    - debias_span (single span debiasing)
    - debias_batch (multiple spans)
    - marker_generator (new marker creation)
    \"\"\"

    def __init__(self, templates_path: Optional[Path] = None):
        \"\"\"Initialize the prompt manager.

        Args:
            templates_path: Path to YAML templates file. If None, uses built-in templates.
        \"\"\"
        self.templates: Dict[str, PromptTemplate] = {}
        self.templates_path = templates_path

        # Load templates
        if templates_path and templates_path.exists():
            self._load_templates_from_file(templates_path)
        else:
            self._load_builtin_templates()

    def _load_templates_from_file(self, path: Path) -> None:
        \"\"\"Load templates from YAML file.\"\"\"
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)

            for name, template_data in data.items():
                self.templates[name] = PromptTemplate(
                    name=name,
                    role=template_data.get('role', 'user'),
                    content=template_data['content'],
                    variables=self._extract_variables(template_data['content']),
                    description=template_data.get('description')
                )

        except Exception as e:
            logger.error(f\"Failed to load templates from {path}: {e}\")
            raise PromptError(f\"Failed to load templates: {e}\")

    def _load_builtin_templates(self) -> None:
        \"\"\"Load built-in templates based on the specification.\"\"\"
        builtin_templates = {
            \"debiaser_system\": {
                \"role\": \"system\",
                \"content\": \"\"\"Du bist ein hochpräziser Debiasing- und Rewriting-Assistent
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
- Du erzeugst keine epistemischen Präfixe wie \"Faktisch korrekt...\",
  \"Logisch scheint mir...\", \"Rein subjektiv...\". Das macht ein separates
  Self-Bias-Check-Modul nachgelagert.
- Schreibe deine Antworten in der Zielsprache, angegeben durch {{output_language}}.\"\"\",
                \"description\": \"System prompt for LLM debiasing assistant\"
            },

            \"debias_span\": {
                \"role\": \"user\",
                \"content\": \"\"\"Du sollst eine konkrete Textpassage debiasen und zwei Alternativen formulieren.

SPRACHE UND KONTEXT
- Originalsprache: {{input_language}}
- Zielsprache der Antwort: {{output_language}}
- Senderkultur (Autor:in): {{sender_culture}}
- Empfängerkultur (Adressat:in): {{receiver_culture}}
- Kontext/Thema: {{context_topic}}
- Zielgruppe / Setting: {{audience}}
- Formalitätsgrad: {{formality_level}}

MARKIERTE PASSAGE
- Voller Originalsatz / Abschnitt:
  \"\"\"{{full_sentence_or_paragraph}}\"\"\"

- Markierter Bias-Span (die problematische Stelle):
  \"\"\"{{bias_span}}\"\"\"

BIAS-METADATEN
- Bias-Familie: {{bias_family}}
- Bias-Subtyp: {{bias_subtype}}
- Schweregrad (0–10) vor Kultur-Anpassung: {{severity_raw}}
- Schweregrad (0–10) nach Kultur-Anpassung:
  - Senderkultur: {{severity_sender}}
  - Empfängerkultur: {{severity_receiver}}
- Kurzbegründung der Cultural Engine:
  \"\"\"{{cultural_explanation}}\"\"\"

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
- Kein \"Tone Policing\": Du darfst Emotion ausdrücken, solange sie nicht abwertend
  oder herabwürdigend gegenüber einer Gruppe ist.
- Schreibe alle Texte in {{output_language}}.

OUTPUT-FORMAT (STRICT JSON)
Antworte ausschließlich mit einem einzigen JSON-Objekt, ohne weitere Kommentare:

{
  \"span_id\": \"{{span_id}}\",
  \"language\": \"{{output_language}}\",
  \"bias_family\": \"{{bias_family}}\",
  \"bias_subtype\": \"{{bias_subtype}}\",
  \"analysis_explanation\": \"…kurze Erklärung in {{output_language}}…\",
  \"can_preserve_core_intent\": true,
  \"variant_A_rewrite\": \"…Variante A, neutral & sachlich…\",
  \"variant_B_rewrite\": \"…Variante B, emotional aber bias-reduziert…\",
  \"safety_notes\": \"…Hinweise, z.B. falls ursprüngliche Intention extrem toxisch war…\"
}\"\"\",
                \"description\": \"Prompt for debiasing a single text span\"
            },

            \"debias_batch\": {
                \"role\": \"user\",
                \"content\": \"\"\"Du sollst mehrere markierte Bias-Spans in einem Dokument debiasen und jeweils zwei Alternativen formulieren.

GLOBALER KONTEXT
- Dokument-Sprache (Original): {{input_language}}
- Zielsprache der Antwort: {{output_language}}
- Senderkultur (Autor:in): {{sender_culture}}
- Empfängerkultur (Adressat:in): {{receiver_culture}}
- Kontext/Thema: {{context_topic}}
- Zielgruppe / Setting: {{audience}}
- Formalitätsgrad: {{formality_level}}

HIER IST DAS ORIGINALDOKUMENT (KONTEXT):
\"\"\"{{full_document_text}}\"\"\"

HIER IST DIE LISTE DER MARKIERTEN SPANS ALS JSON-ARRAY:
{{spans_json}}

DEINE AUFGABEN
- Gehe jeden Eintrag im Array nacheinander durch.
- Führe die gleiche Analyse durch wie im Single-Span-Fall:
  - 1–3 Sätze Erklärung, warum problematisch (inkl. Kulturkontext).
  - Variante A (neutral & sachlich)
  - Variante B (emotional, aber bias-reduziert)
- Überschneidungen:
  - Wenn mehrere Spans das gleiche Bias-Motiv haben, kannst du das in den
    \"safety_notes\" kurz erwähnen (z. B. „wiederholtes Othering einer Gruppe\").

OUTPUT-FORMAT (STRICT JSON)
Antworte ausschließlich mit einem einzigen JSON-Objekt:

{
  \"language\": \"{{output_language}}\",
  \"spans\": [
    {
      \"span_id\": \"s1\",
      \"bias_family\": \"...\",
      \"bias_subtype\": \"...\",
      \"analysis_explanation\": \"...\",
      \"can_preserve_core_intent\": true,
      \"variant_A_rewrite\": \"...\",
      \"variant_B_rewrite\": \"...\",
      \"safety_notes\": \"...\"
    }
  ]
}\"\"\",
                \"description\": \"Prompt for batch debiasing multiple spans\"
            },

            \"marker_generator\": {
                \"role\": \"user\",
                \"content\": \"\"\"Du sollst neue, bias-bereinigte Marker für eine Bias-Kategorie erzeugen.

SPRACHE UND KONTEXT
- Zielsprache: {{output_language}}
- Anwendungsbereich: {{domain}}
- Kulturkontext (typischer Nutzerkreis): {{primary_cultures}}

BIAS-KATEGORIE
- Bias-Familie: {{bias_family}}
- Bias-Subtyp: {{bias_subtype}}
- Kurzbeschreibung: \"\"\"{{bias_description}}\"\"\"

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
- severity_hint: ein Bereich, z. B. \"3-5\", \"7-9\".
- languages: Liste der unterstützten Sprachen, z. B. [\"de\", \"en\"].

WICHTIG
- In den Beispielen kannst du problematische Inhalte zeigen, aber nur so weit, wie es für
  das Erkennen des Musters nötig ist.
- Erfinde keine zusätzlichen slurs oder besonders brutale Formulierungen.
- Nutze möglichst generische oder leicht entschärfte Formulierungen, die trotzdem klar sind.

OUTPUT-FORMAT (STRICT JSON)
Antworte ausschließlich mit einem einzigen JSON-Objekt:

{
  \"bias_family\": \"{{bias_family}}\",
  \"bias_subtype\": \"{{bias_subtype}}\",
  \"language\": \"{{output_language}}\",
  \"markers\": [
    {
      \"id\": \"marker_id_1\",
      \"name\": \"Kurzer Markername\",
      \"description\": \"…\",
      \"rationale\": \"…\",
      \"positive_examples\": [
        \"Beispiel 1 …\",
        \"Beispiel 2 …\",
        \"Beispiel 3 …\"
      ],
      \"counter_example\": \"…\",
      \"severity_hint\": \"7-9\",
      \"languages\": [\"de\", \"en\"]
    }
  ]
}\"\"\",
                \"description\": \"Prompt for generating new bias markers\"
            },

            \"self_bias_check\": {
                \"role\": \"user\",
                \"content\": \"\"\"Du sollst den folgenden Text auf epistemische Selbst-Bias prüfen und korrigieren.

TEXT ZUR PRÜFUNG:
\"\"\"{{text}}\"\"\"

KONTEXT:
{{context}}

AUFGABEN:
1) Klassifiziere den Text epistemisch:
   - \"faktisch\": Objektive, überprüfbare Aussagen
   - \"logisch\": Rationale Schlussfolgerungen und Argumente
   - \"subjektiv\": Meinungen, persönliche Einschätzungen

2) Prüfe auf Überlegenheitsgefälle (Overconfidence):
   - Werden unsichere Aussagen als sicher präsentiert?
   - Werden Meinungen als Fakten dargestellt?
   - Fehlen wichtige Einschränkungen oder Unsicherheitsmarker?

3) Erkenne Bias-Indikatoren:
   - Absolute Formulierungen ohne Belege
   - Verallgemeinerungen
   - Emotionale Sprache bei faktischen Aussagen
   - Fehlende Quellenangaben bei Fakten-Claims

4) Korrigiere mit epistemischem Präfix:
   - \"Faktisch korrekt sage ich, dass...\" für objektive Fakten
   - \"Logisch scheint mir, dass...\" für rationale Argumente
   - \"Rein subjektiv, aus meinem Denken ergibt sich...\" für Meinungen

OUTPUT-FORMAT (STRICT JSON):
{
  \"original_text\": \"...\",
  \"epistemic_classification\": \"faktisch|logisch|subjektiv\",
  \"overconfidence_detected\": true/false,
  \"bias_indicators\": [\"...\", \"...\"],
  \"corrected_text\": \"[Präfix] [korrigierter Text]\",
  \"confidence_score\": 0.0-1.0,
  \"explanation\": \"Begründung der Klassifikation und Korrektur\"
}\"\"\",
                \"description\": \"Prompt for self-bias checking with epistemic classification\"
            }
        }

        for name, template_data in builtin_templates.items():
            self.templates[name] = PromptTemplate(
                name=name,
                role=template_data['role'],
                content=template_data['content'],
                variables=self._extract_variables(template_data['content']),
                description=template_data.get('description')
            )

    def _extract_variables(self, content: str) -> List[str]:
        \"\"\"Extract variable names from template content.\"\"\"
        # Find all {{variable_name}} patterns
        pattern = r'\\{\\{([^}]+)\\}\\}'
        matches = re.findall(pattern, content)
        return list(set(matches))

    def get_template(self, name: str) -> PromptTemplate:
        \"\"\"Get a template by name.\"\"\"
        if name not in self.templates:
            raise TemplateNotFoundError(f\"Template '{name}' not found\")
        return self.templates[name]

    def list_templates(self) -> List[str]:
        \"\"\"List all available template names.\"\"\"
        return list(self.templates.keys())

    def render_template(
        self,
        template_name: str,
        variables: Dict[str, Any],
        strict: bool = True
    ) -> str:
        \"\"\"
        Render a template with provided variables.

        Args:
            template_name: Name of the template
            variables: Dictionary of variable values
            strict: If True, raises error for missing variables

        Returns:
            Rendered template content

        Raises:
            TemplateNotFoundError: If template doesn't exist
            VariableError: If required variables are missing
        \"\"\"
        template = self.get_template(template_name)

        # Check for missing variables
        missing_vars = set(template.variables) - set(variables.keys())
        if missing_vars and strict:
            raise VariableError(f\"Missing required variables: {missing_vars}\")

        # Validate variable types
        for var_name, var_value in variables.items():
            if var_value is None:
                variables[var_name] = \"\"
            elif not isinstance(var_value, (str, int, float, bool)):
                # Convert complex objects to string representation
                if isinstance(var_value, (list, dict)):
                    variables[var_name] = self._serialize_complex_value(var_value)
                else:
                    variables[var_name] = str(var_value)

        try:
            # Use Template for safe substitution
            template_obj = Template(template.content)
            return template_obj.safe_substitute(variables)
        except Exception as e:
            raise VariableError(f\"Template rendering failed: {e}\")

    def _serialize_complex_value(self, value: Union[List, Dict]) -> str:
        \"\"\"Serialize complex values for template substitution.\"\"\"
        if isinstance(value, dict):
            return yaml.dump(value, default_flow_style=False, allow_unicode=True)
        elif isinstance(value, list):
            if all(isinstance(item, str) for item in value):
                return ', '.join(value)
            else:
                return yaml.dump(value, default_flow_style=False, allow_unicode=True)
        return str(value)

    def create_message(
        self,
        template_name: str,
        variables: Dict[str, Any],
        strict: bool = True
    ) -> Dict[str, str]:
        \"\"\"
        Create a message dictionary for LLM consumption.

        Args:
            template_name: Name of the template
            variables: Template variables
            strict: Whether to enforce all variables

        Returns:
            Dictionary with 'role' and 'content' keys
        \"\"\"
        template = self.get_template(template_name)
        content = self.render_template(template_name, variables, strict)

        return {
            \"role\": template.role,
            \"content\": content
        }

    def validate_template(self, template_name: str) -> bool:
        \"\"\"Validate that a template exists and is properly formatted.\"\"\"
        try:
            template = self.get_template(template_name)
            # Try to render with empty variables to check syntax
            self.render_template(template_name, {}, strict=False)
            return True
        except Exception as e:
            logger.error(f\"Template validation failed for '{template_name}': {e}\")
            return False

    def add_template(self, template: PromptTemplate) -> None:
        \"\"\"Add a new template to the manager.\"\"\"
        self.templates[template.name] = template

    def remove_template(self, name: str) -> None:
        \"\"\"Remove a template by name.\"\"\"
        if name in self.templates:
            del self.templates[name]

    def export_templates(self, path: Path) -> None:
        \"\"\"Export all templates to a YAML file.\"\"\"
        templates_data = {}
        for name, template in self.templates.items():
            templates_data[name] = {
                \"role\": template.role,
                \"content\": template.content,
                \"description\": template.description
            }

        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(templates_data, f, default_flow_style=False, allow_unicode=True)