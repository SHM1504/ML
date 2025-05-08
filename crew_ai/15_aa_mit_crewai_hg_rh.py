
"""
Created on Mon Apr 28 15:24:51 2025

@author: Hendrik Gafert und Ralph Herrmann

Source: https://github.com/joaomdmoura/crewAI

Requirements:
    - conda create -n crewai spyder=5.5.1 python=3.12 -c conda-forge
    - pip install crewai crewai-tools duckduckgo-search langchain
    - install Ollama from https://ollama.com/
    - download LLMs (z.B. mistral, llama3.2, openhermes) from https://ollama.com/library
"""

from crewai import Agent, Task, Crew, LLM
    # Importieren der zentralen CrewAI-Komponenten
from crewai.tools import tool  
    # Dekorator, um eigene Tools für Agenten zu erstellen
from langchain.tools import DuckDuckGoSearchRun  
    # Fertiges Tool für Websuche mit DuckDuckGo


# ---------------------------------------
# 1. Einleitung
# ---------------------------------------

# Autonome Agenten haben jeder:
    
# - eine Aufgabe (was sollen sie erreichen),
# - eine Wahrnehmung (was wissen sie über die Umgebung oder Daten),
# - eine Entscheidungslogik (wie sie entscheiden, was zu tun ist),
# - und oft eine Möglichkeit zu handeln (z.B. Nachrichten schreiben, Daten verarbeiten, APIs aufrufen).


# CrewAI basiert auf dem Prinzip, dass jeder Agent das LLM nutzt, um:
    
# - Aufgaben zu verstehen und auszuführen,
# - mit anderen Agenten zu kommunizieren,
# - Texte zu schreiben, zusammenzufassen, zu recherchieren usw.

# CrewAI kümmert sich automatisch darum, dass die Informationen und Aufgaben 
# von einem Agenten an den nächsten weitergegeben werden.
    

# ---------------------------------------
# 2. LLM auswählen und (Thema definieren)
# ---------------------------------------

# Das Thema, zu dem die Agenten arbeiten sollen:
    
# topic = "Aktuelle Trends in KI"

# Auswahl und Konfiguration des Sprachmodells (hier lokal über Ollama)
ollama_llm = LLM(
    model="ollama/llama3.2",  
        # Name des lokal installierten Modells
    temperature=0.0)          
        # 0.0 bedeutet sehr deterministische Antworten (weniger Kreativität).
        # Keine Zufallselemente wie kreative Umschreibungen oder alternative Formulierungen.
        # Das Modell folgt strikt der wahrscheinlichsten Wortwahl.


# --------------------------------------------
# 3. Tool definieren: Websuche über DuckDuckGo
# --------------------------------------------

@tool('DuckDuckGoSearch')
    # @tool-Dekorator markiert die Funktion als CrewAI-Tool
    # Die Agenten können das Tool aktiv verwenden, wenn sie Aufgaben erledigen
    # Der Name des Tools ist 'DuckDuckGoSearch'
def search_tool(search_query: str) -> str:
    """
    Tool zur Durchführung einer Websuche mit DuckDuckGo.
    
    Diese Funktion wird den Agenten zur Verfügung gestellt,
    um aktuelle Informationen aus dem Internet abzurufen.

    Args:
        search_query (str): Der Suchbegriff oder die Fragestellung 
            # Der Eingabeparameter ist vom Typ 'str' also Text            

    Returns:
        str: Ergebnisse der Websuche als Text.
            # Der Rückgabewert ist vom Typ 'str' also Text 
    """
    return DuckDuckGoSearchRun().run(search_query)
        # DuckDuckGoSearchRun() ist eine LangChain-Toolklasse
        # .run() führt eine Live-Suche bei DuckDuckGo aus und gibt Treffer zurück
        # Vorteil: Schnell, einfach, keine API-Keys nötig


# ---------------------------------------
# 4. Agenten definieren
# ---------------------------------------

# Was passiert hier konkret:
    
# 1.	Manager bekommt als erstes Überblick über die Aufgaben.
# 2.	Manager entscheidet, welcher Agent was tun soll.
# 3.	Forscher arbeitet seine Research-Task ab und liefert eine strukturierte Liste.
# 4.	Schreiber bekommt diese Liste automatisch als Input und baut daraus einen Bericht.
# 5.	CrewAI steuert die Kommunikation, Aufgabenübergabe und Reihenfolge vollautomatisch.

# Hinweis: Bei dieser Struktur kannst du in CrewAI sogar Regeln hinzufügen, z.B. Feedback-Schleifen, 
# Validierungen oder Retry-Strategien, wenn du später noch professioneller arbeiten willst.

# Parameter bei der Definition von Agenten:
    
# manager = Agent(
#     role="Projektmanager", # Titel/Funktion des Agenten
#     goal="...",            # Ziel: Was soll der Agent erreichen / Was ist seine Aufgabe?
#     backstory="...",       # Hintergrund: Warum ist er qualifiziert?
#     llm=ollama_llm,        # Welches LLM-Modell nutzt der Agent? Hier: llama3.2
#     verbose=True)          # True = zeige Zwischenschritte beim Arbeiten

# Projektmanager koordiniert die Crew und behält Qualität im Auge:
manager = Agent(
    role="Projektmanager",
    goal="Koordiniere die Erstellung und Qualitätskontrolle eines Berichts über aktuelle KI-Trends",
    backstory="Ein erfahrener Manager, der Teams effizient organisiert und hohe Qualitätsstandards durchsetzt.",
    llm=ollama_llm,
    verbose=True)

# Forscher sucht aktuelle Informationen im Internet:
researcher = Agent(
    role="Forscher",
    goal="Finde aktuelle Trends in der Künstlichen Intelligenz",
    backstory="Ein neugieriger Analyst, der sich auf technologische Entwicklungen spezialisiert hat.",
    llm=ollama_llm,
    verbose=True)

# Schreiber formuliert die Inhalte klar und strukturiert:
writer = Agent(
    role="Schreiber",
    goal="Schreibe einen Bericht basierend auf den Forschungsergebnissen",
    backstory="Ein talentierter Redakteur, der komplexe Inhalte klar und strukturiert formulieren kann.",
    llm=ollama_llm,
    verbose=True)

# Editor verfeinert den Text sprachlich und stilistisch:
editor = Agent(
    role="Editor",
    goal="Verbessere den Stil, die Lesbarkeit und die sprachliche Eleganz des Berichts",
    backstory="Ein erfahrener Lektor mit einem Gespür für klare, präzise und elegante Ausdrucksweise.",
    llm=ollama_llm,
    verbose=True)


# ---------------------------------------
# 5. Aufgaben (Tasks) definieren
# ---------------------------------------

# Parameter bei der Vergabe der Aufgaben:
    
# research_task = Task(
#     description=...,        # Was soll gemacht werden? (Aufgabenbeschreibung)
#     expected_output=...,    # Wie soll das Ergebnis aussehen?
#     agent=researcher)       # Welcher Agent führt die Aufgabe aus?

# Forschungsaufgabe: Finden und Zusammenfassen aktueller KI-Trends
research_task = Task(
    description=(
        "Untersuche aktuelle Trends im Bereich der künstlichen Intelligenz "
        "und fasse fünf wichtige Trends in kurzen Stichpunkten zusammen."),
    expected_output="Eine Liste von fünf aktuellen KI-Trends mit je 2-3 kurzen Erklärungen.",
    agent=researcher)

# Schreibaufgabe: Erstellung eines vollständigen Berichts basierend auf den Trends:
writing_task = Task(
    description=(
        "Nutze die Forschungsergebnisse, um einen zusammenhängenden Bericht "
        "über KI-Trends zu schreiben. Der Bericht soll ca. 300 Wörter umfassen "
        "und für ein technikinteressiertes Publikum verständlich sein."),
    expected_output="Ein Bericht mit Einleitung, fünf Abschnitten zu den Trends und einem Fazit.",
    agent=writer)

# Lektoratsaufgabe: Feinschliff des Berichts für bessere Lesbarkeit:
editing_task = Task(
    description=(
        "Optimiere den Bericht, indem du den Schreibstil verfeinerst, "
        "Satzstrukturen klarer formulierst und die Lesbarkeit erhöhst. "
        "Achte auf Sprachfluss, Prägnanz und stilistische Eleganz."),
    expected_output="Eine stilistisch verbesserte Version des Berichts, flüssig und ansprechend formuliert.",
    agent=editor)


# ---------------------------------------
# 6. Crew erstellen
# ---------------------------------------

# Crew = Zusammenstellung der Agenten und Aufgaben
                # (optional Feedback-Management durch Manager)
crew = Crew(
    agents=[manager, researcher, writer, editor],       # Team-Mitglieder
    tasks=[research_task, writing_task, editing_task],  # Aufgaben
    manager=manager,                                    # Manager überwacht den Projektablauf
    verbose=True,                                       # Ausgabe aller Zwischenschritte im Terminal
#    process_feedback=True,                             # aktiviert Feedback-Mechanismus des Managers
#    retry_failed_tasks=True,                           # automatische Wiederholung bei schlechter Qualität
#    max_retries=2                                      # max. zwei Verbesserungsrunden
)

# --------------------------------------------
# 7. Crew starten (Projektbeginn) und ausgeben
# --------------------------------------------

result = crew.kickoff()
    # kickoff() startet den gesamten Ablauf: 
    # Die Agenten führen ihre Aufgaben der Reihe nach aus

# Ergebnis anzeigen
print("\n--- Endergebnis der Crew nach Lektorat ---\n")
print(result)


# ---------------------------------------
# 8. Erklärungen
# ---------------------------------------

# Was ist Ollama?
# Ollama ist eine Software-Plattform, die es extrem einfach macht, große Sprachmodelle (LLMs) 
# lokal auf deinem eigenen Computer laufen zu lassen. Es ist eine super bequeme Möglichkeit, 
# Open-Source-Modelle wie Mistral, Llama 3 oder OpenHermes auf deinem eigenen Laptop oder Server 
# zu betreiben – ohne komplizierte Installation.

# Was ist LangChain?
# LangChain ist ein Open-Source-Framework, das speziell dafür entwickelt wurde, Sprachmodelle (LLMs)
# wie GPT, LLaMA, Claude, etc. mit externen Datenquellen und Funktionen zu kombinieren.

# Was ist DuckDuckGo?
# DuckDuckGo ist eine datenschutzfreundliche Internet-Suchmaschine, die sich von Google, Bing & Co. 
# vor allem durch Privatsphäre und Werbefreiheit unterscheidet.

# Was ist DuckDuckGoSearchRun?
# DuckDuckGoSearchRun ist eine LangChain-Toolklasse, die es einem Sprachmodell (LLM) ermöglicht, 
# aktuelle Websuchen über die Suchmaschine DuckDuckGo durchzuführen.


# ---------------------------------------
# 9. Aufgaben
# ---------------------------------------

# Einbinden eines anderen LLM's und Vergleich der Ergebnisse. 
# Einbinden weiterer Agenten (z.B. Hacker)
