import os

from openai import OpenAI


def get_openai_client() -> OpenAI:
    if not os.environ.get("OPENAI_API_KEY"):
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. "
            "Add it to a .env file or export it before using LLM features."
        )
    return OpenAI()
