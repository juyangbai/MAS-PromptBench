#!/usr/bin/env python
import warnings
from datetime import datetime

from topologies.sequential.crewai.crewai_base.crew import SequentialResearchCrew

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")


def run():
    inputs = {
        "topic": "AI agent topologies",
        "current_year": str(datetime.now().year),
    }
    SequentialResearchCrew().crew().kickoff(inputs=inputs)


if __name__ == "__main__":
    run()
