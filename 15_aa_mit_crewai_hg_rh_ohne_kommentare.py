
"""
Created on Mon Apr 28 15:24:51 2025
@author: Hendrik Gafert und Ralph Herrmann
Source: https://github.com/joaomdmoura/crewAI
"""

from crewai import Agent, Task, Crew, LLM
from crewai.tools import tool  
from langchain.tools import DuckDuckGoSearchRun  

ollama_llm = LLM(model="ollama/llama3.2", temperature=0.0)          

@tool('DuckDuckGoSearch')
def search_tool(search_query: str) -> str:
    """
    Tool zur Durchführung einer Websuche mit DuckDuckGo
    """
    return DuckDuckGoSearchRun().run(search_query)
       

manager = Agent(
    role="Projektmanager",
    goal="Koordiniere die Erstellung und Qualitätskontrolle eines Berichts über aktuelle KI-Trends",
    backstory="Ein erfahrener Manager, der Teams effizient organisiert und hohe Qualitätsstandards durchsetzt.",
    llm=ollama_llm,
    verbose=True)

researcher = Agent(
    role="Forscher",
    goal="Finde aktuelle Trends in der Künstlichen Intelligenz",
    backstory="Ein neugieriger Analyst, der sich auf technologische Entwicklungen spezialisiert hat.",
    llm=ollama_llm,
    verbose=True)

writer = Agent(
    role="Schreiber",
    goal="Schreibe einen Bericht basierend auf den Forschungsergebnissen",
    backstory="Ein talentierter Redakteur, der komplexe Inhalte klar und strukturiert formulieren kann.",
    llm=ollama_llm,
    verbose=True)

editor = Agent(
    role="Editor",
    goal="Verbessere den Stil, die Lesbarkeit und die sprachliche Eleganz des Berichts",
    backstory="Ein erfahrener Lektor mit einem Gespür für klare, präzise und elegante Ausdrucksweise.",
    llm=ollama_llm,
    verbose=True)

research_task = Task(description=(
        "Untersuche aktuelle Trends im Bereich der künstlichen Intelligenz "
        "und fasse fünf wichtige Trends in kurzen Stichpunkten zusammen."),
    expected_output="Eine Liste von fünf aktuellen KI-Trends mit je 2-3 kurzen Erklärungen.",
    agent=researcher)

writing_task = Task(description=(
        "Nutze die Forschungsergebnisse, um einen zusammenhängenden Bericht "
        "über KI-Trends zu schreiben. Der Bericht soll ca. 300 Wörter umfassen "
        "und für ein technikinteressiertes Publikum verständlich sein."),
    expected_output="Ein Bericht mit Einleitung, fünf Abschnitten zu den Trends und einem Fazit.",
    agent=writer)

editing_task = Task(description=(
        "Optimiere den Bericht, indem du den Schreibstil verfeinerst, "
        "Satzstrukturen klarer formulierst und die Lesbarkeit erhöhst. "
        "Achte auf Sprachfluss, Prägnanz und stilistische Eleganz."),
    expected_output="Eine stilistisch verbesserte Version des Berichts, flüssig und ansprechend formuliert.",
    agent=editor)

crew = Crew(
    agents=[manager, researcher, writer, editor], tasks=[research_task, writing_task, editing_task],  
    manager=manager, verbose=True, process_feedback=True, retry_failed_tasks=True, max_retries=2)

result = crew.kickoff()

print("\n--- Endergebnis der Crew nach Lektorat ---\n")
print(result)

