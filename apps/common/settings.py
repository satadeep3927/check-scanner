from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    github_copilot_token: str = Field(default="")
    llm_model: str = Field(default="gpt-4.1")
    llm_api_base_url: str = Field(default="https://api.githubcopilot.com/")
    copilot_extra_headers: dict = Field(
        default={
            "editor-version": "vscode/1.104.0",
            "editor-plugin-verion": "copilot.vim/1.16.0",
            "user-agent": "GithubCopilot/1.155.0",
            "Copilot-Vision-Request": "true",
        }
    )

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
    }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
