from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from dotenv import load_dotenv
from os import getenv
from loguru import logger
from openai import OpenAI
from openai.types.responses import Response

load_dotenv()


class OpenAIClient:
    """
    Client to Contact OpenAI we deployed in Azure Foundry Agents
    to respond when prompted with a query, it has already pre-instructed to summarize YouTube videos when provided
    """
    openai_client: OpenAI

    def __init__(self):
        self.endpoint = getenv("AZURE_URL")
        self.project_client = AIProjectClient(
            endpoint=self.endpoint,
            credential=DefaultAzureCredential(),
        )
        self.agent = self.project_client.agents.get(agent_name=getenv("AZURE_AGENT"))
        self.openai_client = self.project_client.get_openai_client()
        logger.info(f"Retrieved agent: {self.agent.name}")

    def __enter__(self):
        self.openai_client = self.project_client.get_openai_client()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.openai_client.close()
        self.project_client.close()

    def ask(self, prompt: str) -> Response:
        response = self.openai_client.responses.create(
            input=[{"role": "user", "content": prompt}],
            extra_body={"agent": {"name": self.agent.name, "type": "agent_reference"}},
        )
        return response

    @classmethod
    def parse_response(cls, response: Response) -> str:
        return response.output_text


if __name__ == "__main__":
    with OpenAIClient() as client:
        print(client.ask("Tell me what you can help with."))