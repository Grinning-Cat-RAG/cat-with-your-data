from enum import Enum
from typing import Dict, Any
from pydantic import BaseModel, Field, field_validator

from cat import plugin, log
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
        "conn_str": "mssql+pymssql://{username}:{password}@{host}:{port}/{database}"
    },
    "Microsoft SQL Server ODBC": {
        "agent_type": "sql",
        "conn_str": "mssql+pyodbc://{username}:{password}@{database}"
    },
    # the file datasources are files in the file manager of the agent (``host``: their path in the folder of the agent)
    "SQLite": {
        "agent_type": "sqlite"
    },
    "CSV": {
        "agent_type": "csv"
    },
    "JSON": {
        "agent_type": "json"
    }
}

# Create a dynamic enum for database types; the empty value means "no configured datasource" (only uploaded datasets)
DatasourceType = Enum(
    "DatasourceType", [("NONE", "")] + [(key.replace(" ", "_"), key) for key, _ in datasources.items()]
)


class ChartMode(Enum):
    DISABLED = "disabled"
    ON_REQUEST = "on_request"
    AUTO = "auto"


class SqlAgentType(Enum):
    AUTO = "auto"
    REACT = "react"
    TOOL_CALLING = "tool_calling"


class MySettings(BaseModel):
    ds_type: DatasourceType = Field(
        title="datasource type",
        default=""
    )
    host: str = Field(
        title="host or file path",
        description="SQL databases: the host. CSV, JSON and SQLite: the path of the file in the file manager of the "
                    "agent, relative to the folder of the agent (e.g. sales.csv, a file uploaded in its memory)",
        default=""
    )
    port: int = Field(
        title="port",
        default=0
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
    agent_type: SqlAgentType = Field(
        title="SQL agent type",
        description="auto: tool calling if the LLM supports it, otherwise ReAct; react: text-based ReAct agent "
                    "(works with every LLM); tool_calling: native tool calling (more reliable, needs a chat model "
                    "supporting tools)",
        default=SqlAgentType.AUTO,
    )
    charts: ChartMode = Field(
        title="charts",
        description="disabled: never draw charts; on_request: draw a chart only when the user asks for it; "
                    "auto: draw a chart also when the answer is easier to understand visually",
        default=ChartMode.ON_REQUEST,
    )
    chart_max_rows: int = Field(
        title="maximum number of rows fetched to draw a chart",
        default=1000,
        ge=1,
    )
    thought_max_rows: int = Field(
        title="maximum number of rows of the chart data shown to the agent",
        default=30,
        ge=1,
    )
    use_uploaded_datasets: bool = Field(
        title="query the uploaded CSV/SQLite datasets instead of the configured datasource, when available",
        default=True,
    )
    capture_rabbithole_uploads: bool = Field(
        title="register the CSV/SQLite files uploaded through the Rabbit Hole as queryable datasets",
        default=True,
    )
    max_upload_size_mb: int = Field(
        title="maximum size (MB) of an uploaded dataset",
        default=100,
        ge=1,
    )
    chat_datasets_ttl_hours: int = Field(
        title="hours after which the datasets of an idle conversation are deleted (0 = never)",
        default=72,
        ge=0,
    )

    # file datasources have no port: the examples (and the settings saved by previous versions) leave it empty
    @field_validator("port", mode="before")
    @classmethod
    def _empty_port(cls, value):
        return 0 if value in ("", None) else value


def _decrypted(stored: Dict[str, Any], agent_id: str) -> Dict[str, Any]:
    settings, failed = decrypt_secrets(stored, StringCrypto())
    for key in failed:
        log.error(
            f"[connectors] agent {agent_id}: cannot decrypt '{key}' (was CAT_CRYPTO_KEY changed?): "
            "it is ignored until saved again"
        )
    return settings


def _with_defaults(settings: Dict[str, Any]) -> Dict[str, Any]:
    """Stored settings completed with the defaults of the fields added in later versions."""
    return {**MySettings().model_dump(mode="json"), **(settings or {})}


@plugin
def settings_schema():
    return MySettings.model_json_schema()


@plugin
async def load_settings(plugin_id: str, agent_id: str) -> Dict[str, Any]:
    stored = await crud_plugins.get_setting(agent_id, plugin_id)
    if stored is None:
        return MySettings().model_dump(mode="json")
    return _with_defaults(_decrypted(stored, agent_id))


@plugin
async def save_settings(plugin_id: str, settings: Dict[str, Any], agent_id: str) -> Dict[str, Any]:
    stored = await crud_plugins.update_setting(agent_id, plugin_id, encrypt_secrets(settings, StringCrypto()))
    return _with_defaults(_decrypted(stored, agent_id))
