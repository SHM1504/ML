#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 15:24:51 2024

@author: d

Source: https://github.com/joaomdmoura/crewAI

Description: test for git

requirements: conda create -n crewai spyder python=3.12 -c conda-forge
              pip install crewai crewai-tools duckduckgo-search langchain
              
              install Ollama from https://ollama.com/
              download LLM (e.g. mistral, llama3.2, openhermes),
                  link: https://ollama.com/library
"""

from crewai import Agent, Task, Crew, LLM, Process
from crewai.tools import tool
from langchain.tools import DuckDuckGoSearchRun # , ShellTool

# Define the topic
topic = "crypto market"

# Define the LLM model (e.g., codellama, llama3.2, magicoder, mistral, openhermes)
ollama_llm = LLM(model="ollama/llama3.2", temperature=0.0)
# ollama_llm = LLM(model="ollama/llama3.2", temperature=0.0) # not good at using tools

# Define tools
# shell_tool = ShellTool()
# shell_tool.description = shell_tool.description + f"args {shell_tool.args}".replace(
#     "{", "{{"
# ).replace("}", "}}")

@tool('DuckDuckGoSearch')
def search_tool(search_query: str) -> str:
    """
    Search the web for information on a given topic.

    Args:
        search_query (str): The search query string to look up information.

    Returns:
        str: Search results from DuckDuckGo.
    """
    return DuckDuckGoSearchRun().run(search_query)

# Define agents with roles and goals
researcher = Agent(
    role='Senior Research Analyst',
    goal=f'Uncover cutting-edge developments in the {topic} sector',
    backstory="""You work at a leading tech think tank, specializing in
    identifying emerging trends and presenting actionable insights through
    complex data analysis.""",
    verbose=True,
    allow_delegation=False,
    tools=[search_tool],
    llm=ollama_llm
)

writer = Agent(
    role='Tech Content Strategist',
    goal='Craft compelling content on tech advancements',
    backstory="""As a renowned Content Strategist, you are known for
    transforming complex concepts into engaging and insightful narratives.""",
    verbose=True,
    allow_delegation=False,
    llm=ollama_llm
)

formatter = Agent(
    role='Data Formatter',
    goal='Format data according to defined needs',
    backstory="""You are an expert in data formatting, ensuring input data is 
    transformed into well-structured output.""",
    verbose=True,
    allow_delegation=False,
    llm=ollama_llm
)

# hacker = Agent(
#     role='Local File System Manager',
#     goal='Organize files in the local file system',
#     backstory="""As an organized file system manager, you manage local files 
#     in the current working directory efficiently.""",
#     verbose=True,
#     allow_delegation=False,
#     tools=[shell_tool],
#     llm=ollama_llm
# )

# Create tasks for agents
task1 = Task(
    description=f"""Conduct a comprehensive analysis of the most useful trends
    in the {topic} sector. Identify the most promising trends that yield the
    highest return on investment. Your final output MUST be a detailed analysis
    report. Don't use the search tool more than 5 times.""",
    agent=researcher,
    expected_output='A comprehensive analysis report'
)

task2 = Task(
    description=f"""Develop an engaging blog post using the provided insights, 
    highlighting the most promising developments in the {topic} sector. Ensure
    the content is informative yet accessible, with bullet points catering
    to a tech-savvy audience. Avoid overly complex language. Your
    final output MUST be a full blog post with at least 5 paragraphs.""",
    agent=writer,
    expected_output='An engaging blog post with bullet points and with at least 5 paragraphs'
)

task3 = Task(
    description="""Format the provided blog post into markdown format. Your
    final output MUST be the full blog post formatted as markdown, ready for
    saving as a file.""",
    agent=formatter,
    expected_output='A blog post formatted in markdown'
)

# task4 = Task(
#     description=f"""Save the provided markdown formatted blog post to a file
#     named '{topic.replace(" ", "_")}.md' in the current working directory by
#     using the shell_tool. When using the shell_tool be sure to use the
#     "commands" keyword when using. Do NOT use the "command" keyword.""",
#     agent=hacker,
#     expected_output='A markdown file saved in the current working directory'
# )

# Instantiate the crew with a sequential process
crew = Crew(
    agents=[researcher, writer, formatter], # , hacker
    tasks=[task1, task2, task3], # , task4
    verbose=True,
    process=Process.sequential  # New Processes coming in the future
)

# Get the crew to work!
result = crew.kickoff()

print("######################")
print(result)
