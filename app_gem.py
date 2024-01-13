import os
from langchain_google_genai import ChatGoogleGenerativeAI
from crewai import Agent, Task, Crew, Process
from langchain.tools import DuckDuckGoSearchRun


# Set gemini pro as llm
llm = ChatGoogleGenerativeAI(model="gemini-pro",
                             verbose = True,
                             temperature = 0.5,
                             google_api_key="")


#create searches
tool_search = DuckDuckGoSearchRun()

# Define Agents
email_author = Agent(
    role='Professional Email Author',
    goal='Craft concise and engaging emails',
    backstory='Experienced in writing impactful marketing emails.',
    verbose=True,
    allow_delegation=False,
    llm=llm,
    tools=[
        tool_search
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
