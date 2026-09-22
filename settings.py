from typing import Dict, Any
from pydantic import BaseModel, Field
from enum import Enum

from cat import plugin
from cat.db.cruds import plugins as crud_plugins
from cat.services.string_crypto import StringCrypto

from .crypt import decrypt_secrets, encrypt_secrets


datasources = {
    "PostgreSQL" : {
        "agent_type": "sql",
        "conn_str": "postgresql+psycopg2://{username}:{password}@{host}:{port}/{database}"
    },
    "MySQL": {
        "agent_type": "sql",
        "conn_str": "mysql+mysqlconnector://{username}:{password}@{host}:{port}/{database}"
    },
    "Oracle": {
        "agent_type": "sql",
        "conn_str": "oracle://{username}:{password}@{host}:{port}/{database}"
    },
    "Microsoft SQL Server": {
        "agent_type": "sql",
        "conn_str": "mssql+pymssql://scott:{host}@{host}:{port}/{database}"
    },
    "Microsoft SQL Server ODBC": {
        "agent_type": "sql",
        "conn_str": "mssql+pyodbc://{username}:{password}@{database}"
    },
    "SQLite": {
        "agent_type": "sql",
        "conn_str": "sqlite:///{host}"
    },
    "CSV": {
        "agent_type": "csv"
    },
    "JSON": {
        "agent_type": "json"
    }
}

# Create a dynamic enum for database types
DatasourceType = Enum("DatasourceType", [(key.replace(" ", "_"), key) for key, _ in datasources.items()])

class MySettings(BaseModel):
    ds_type: DatasourceType = Field(
        title="datasource type",
        default=""
    )
    host: str = Field(
        title="host or file path",
        default=""
    )
    port: int = Field(
        title="port",
        default=""
    )
    username: str = Field(
        title="username",
        default=""
    )
    password: str = Field(
        title="password",
        default=""
    )
    database: str = Field(
        title="database",
        default=""
    )
    extra: str = Field(
        title="extra",
        default=""
    )
    input_prompt: str = Field(
        title="input prompt",
        default="""{user_message}""",
    )
    output_prompt: str = Field(
        title="output prompt",
        default="""{prompt_prefix}
You have elaborated the user's question, you have searched for the answer and now you have the solution in your Thought; 
reply to the user briefly, precisely and based on the context of the dialogue.
- Human: {user_message}
- Thought: {thought}
- AI:""",
    )


def _decrypted(stored: Dict[str, Any], agent_id: str) -> Dict[str, Any]:
    settings, failed = decrypt_secrets(stored, StringCrypto())
    for key in failed:
        log.error(
            f"[connectors] agent {agent_id}: cannot decrypt '{key}' (was CAT_CRYPTO_KEY changed?): "
            "it is ignored until saved again"
        )
    return settings


@plugin
def settings_schema():
    return MySettings.model_json_schema()


@plugin
async def load_settings(plugin_id: str, agent_id: str) -> Dict[str, Any]:
    stored = await crud_plugins.get_setting(agent_id, plugin_id)
    if stored is None:
        return MySettings().model_dump()
    return _decrypted(stored, agent_id)


@plugin
async def save_settings(plugin_id: str, settings: Dict[str, Any], agent_id: str) -> Dict[str, Any]:
    stored = await crud_plugins.update_setting(agent_id, plugin_id, encrypt_secrets(settings, StringCrypto()))
    return _decrypted(stored, agent_id)
