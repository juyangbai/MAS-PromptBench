from crewai import Agent, Crew, Process, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, crew, task


@CrewBase
class SequentialResearchCrew:
    """Four-agent sequential pipeline: researcher -> analyst -> writer -> editor."""

    agents: list[BaseAgent]
    tasks: list[Task]

    @agent
    def researcher(self) -> Agent:
        return Agent(config=self.agents_config["researcher"], verbose=True)

    @agent
    def analyst(self) -> Agent:
        return Agent(config=self.agents_config["analyst"], verbose=True)

    @agent
    def writer(self) -> Agent:
        return Agent(config=self.agents_config["writer"], verbose=True)

    @agent
    def editor(self) -> Agent:
        return Agent(config=self.agents_config["editor"], verbose=True)

    @task
    def research_task(self) -> Task:
        return Task(config=self.tasks_config["research_task"])

    @task
    def analysis_task(self) -> Task:
        return Task(config=self.tasks_config["analysis_task"])

    @task
    def writing_task(self) -> Task:
        return Task(config=self.tasks_config["writing_task"])

    @task
    def editing_task(self) -> Task:
        return Task(config=self.tasks_config["editing_task"], output_file="report.md")

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
