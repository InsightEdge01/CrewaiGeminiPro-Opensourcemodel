import os
from crewai import Agent, Task, Crew, Process
from langchain.tools import DuckDuckGoSearchRun


os.environ["HUGGINGFACEHUB_API_TOKEN"] = ""

from langchain_community.llms import HuggingFaceHub
repo_id = "mistralai/Mistral-7B-Instruct-v0.2"  # See https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads for some other options
llm = HuggingFaceHub(
    repo_id=repo_id, model_kwargs={"temperature": 0.5, "max_tokens":4096}
)

search_tool = DuckDuckGoSearchRun()

# Define Agents
email_author = Agent(
    role='Professional Email Author',
    goal='Craft concise and engaging emails',
    backstory='Experienced in writing impactful marketing emails.',
    verbose=True,
    allow_delegation=False,
    llm=llm,
    tools=[
        search_tool
      ]
)
marketing_strategist = Agent(
    role='Marketing Strategist',
    goal='Lead the team in creating effective cold emails',
    backstory='A seasoned Chief Marketing Officer with a keen eye for standout marketing content.',
    verbose=True,
    allow_delegation=True,
    llm=llm
)

content_specialist = Agent(
    role='Content Specialist',
    goal='Critique and refine email content',
    backstory='A professional copywriter with a wealth of experience in persuasive writing.',
    verbose=True,
    allow_delegation=False,
    llm=llm
)

# Define Task
email_task = Task(
    description='''1. Generate two distinct variations of a cold email promoting a video editing solution. 
    2. Evaluate the written emails for their effectiveness and engagement.
    3. Scrutinize the emails for grammatical correctness and clarity.
    4. Adjust the emails to align with best practices for cold outreach. Consider the feedback 
    provided to the marketing_strategist.
    5. Revise the emails based on all feedback, creating two final versions.''',
    agent=marketing_strategist  # The Marketing Strategist is in charge and can delegate
)


# Create a Single Crew
email_crew = Crew(
    agents=[email_author, marketing_strategist, content_specialist],
    tasks=[email_task],
    verbose=True,
    process=Process.sequential
)

# Execution Flow
print("Crew: Working on Email Task")
emails_output = email_crew.kickoff()