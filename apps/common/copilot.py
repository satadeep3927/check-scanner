import cachetools.func as cachetools
import requests
from pydantic import SecretStr

from apps.common.settings import get_settings


@cachetools.ttl_cache(maxsize=1, ttl=600)
def get_access_token_from_copilot() -> SecretStr:
    """
    Retrieves the access token for Copilot from Github API.

    Returns:
        str: The access token if found, otherwise an empty string.
    """

    settings = get_settings()

    response = requests.get(
        "https://api.github.com/copilot_internal/v2/token",
        headers={
            "authorization": f"token {settings.github_copilot_token}",
            "editor-version": "Neovim/0.6.1",
            "editor-plugin-verion": "copilot.vim/1.16.0",
            "user-agent": "GithubCopilot/1.155.0",
        },
        timeout=10,
    )
    if response.status_code != 200:
        raise Exception(
            f"Failed to fetch access token: {response.status_code} - {response.text}"
        )
    payload = response.json()
    token = payload.get("token", "")
    return token
