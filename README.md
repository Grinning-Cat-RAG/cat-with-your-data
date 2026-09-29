# Cat With Your Data

A Grinning Cat plugin that lets you query your data in natural language, getting answers **and charts**.

It can reason over SQL databases, SQLite, CSV and JSON files, and over CSV/SQLite files uploaded on the fly.

<img src="./thumb.jpg" width="400" alt="Cat With Your Data thumbnail" />

## What It Does

- Turns user questions into datasource-aware reasoning steps.
- Draws charts (bar, horizontal bar, line, area, pie, scatter, histogram) when the user asks for them, or
  automatically when a visualization makes the answer clearer.
- Lets users upload CSV and SQLite files on the fly, per conversation or for the whole agent.
- Uses dedicated LangChain agents for SQL (also used for CSV files, tabular JSON files and uploaded datasets) and
  JSON. Charts are drawn by the SQL agent itself, through an extra `draw_chart` tool.
- Supports custom `input_prompt` and `output_prompt` templates.
- Works with multiple SQL engines through SQLAlchemy connection strings.

## Supported Datasources

From `settings.py`, the plugin supports:

- `PostgreSQL`
- `MySQL`
- `Oracle`
- `Microsoft SQL Server`
- `Microsoft SQL Server ODBC`
- `SQLite`
- `CSV`
- `JSON`: tabular JSON files (a list of records, or an object of lists of records) are loaded as tables and queried
  by the SQL agent, so charts are available; the other JSON files are queried by the JSON agent, without charts
- CSV (`.csv`, `.tsv`) and SQLite (`.sqlite`, `.sqlite3`, `.db`, `.db3`) files uploaded on the fly

Reference for SQLAlchemy connection formats:

https://docs.sqlalchemy.org/en/20/core/engines.html

## Requirements

Python dependencies are declared in `pyproject.toml`:

- `langchain-classic`, `langchain-community`
- `matplotlib` (charts)
- `pandas`, `sqlalchemy`, `tabulate==0.9.0`
- `mysql-connector-python`, `psycopg2-binary`

Depending on your datasource, additional DB drivers may be required by SQLAlchemy.

## Configuration

Plugin settings are defined in `settings.py` and loaded at runtime.

Datasource fields:

- `ds_type`: datasource type (for example `PostgreSQL`, `CSV`, `JSON`); it can be left empty if you only work with
  uploaded datasets
- `host`: DB host or local file path for CSV/JSON/SQLite
- `port`: DB port (it can be left empty for file datasources)
- `username`: DB username
- `password`: DB password (stored encrypted)
- `database`: DB name
- `input_prompt`: template used before reasoning (`{user_message}` available)
- `output_prompt`: final response template (`{prompt_prefix}`, `{user_message}`, `{thought}`, `{chat_history}` available)

Agent, charts and uploads fields:

- `agent_type`: type of the LangChain SQL agent. `auto` (default) uses native tool calling when the configured LLM is a
  chat model supporting tools, otherwise ReAct; `react` forces the text-based ReAct agent, which works with every LLM
  but passes the chart arguments as a JSON string and is more error-prone with small models; `tool_calling` forces
  native tool calling (with an LLM without tools the SQL agent cannot be created: the error is logged and the plugin
  does not answer)

- `charts`: `disabled`, `on_request` (default: a chart is drawn only when the user asks for it, in any language) or
  `auto` (a chart is drawn also when comparisons, rankings, trends or distributions are easier to read visually)
- `chart_delivery`: `inline` (default: the chart is embedded in the answer as a base64 markdown image) or `url`
  (the answer contains a markdown image pointing to the plugin endpoint, see below)
- `public_base_url`: public URL of the Cat, used to build the chart URLs when `chart_delivery` is `url`
  (e.g. `https://cat.example.com`); if empty, the URL is relative
- `chart_max_rows`: maximum number of rows fetched to draw a chart
- `thought_max_rows`: maximum number of rows of the chart data shown to the agent after drawing a chart
- `use_uploaded_datasets`: when a conversation has uploaded datasets, query them instead of the configured datasource
- `capture_rabbithole_uploads`: register the CSV/SQLite files uploaded through the Rabbit Hole as datasets
- `max_upload_size_mb`: maximum size of an uploaded dataset
- `chat_datasets_ttl_hours`: hours after which the generated charts, and the datasets of a conversation not uploaded
  nor queried meanwhile, are deleted (`0` = never; shared datasets never expire); the expired ones are removed when a
  dataset is uploaded and when a chart is stored

`chart_max_rows`, `thought_max_rows` and `max_upload_size_mb` must be at least 1, `chat_datasets_ttl_hours` at least 0.

Ready-to-use examples are in `settings_examples/`:

- `settings-postgres.json`
- `settings-mysql.json`
- `settings-csv.json`
- `settings-json.json`
- `settings-prompt.json`
- `settings-examples.json`
- `settings-charts-uploads.json`

## How It Works

1. The hook `agent_fast_reply` (in `query_cat.py`) initializes `QueryCatAgent`.
2. `QueryCatAgent` (in `query_agent.py`) loads settings and picks the datasource: the uploaded datasets, if any,
   otherwise the configured one. SQL datasources, CSV files, tabular JSON files and uploaded datasets are all exposed
   as SQL databases (CSV and JSON files are loaded in an in-memory SQLite database).
3. The LangChain SQL agent (`create_sql_agent`) explores the schema and queries the data with the standard tools of
   `SQLDatabaseToolkit`. When charts are enabled, it also gets the `draw_chart` tool (in `chart_tool.py`), whose
   description tells the agent when to use it according to the `charts` setting. The tool receives a read-only SQL
   query and a chart specification, validates the query (single `SELECT`/`WITH` statement, no data-modifying
   keywords), runs it, renders the chart with matplotlib and returns the plotted data to the agent; on errors, it
   returns the error message, so that the agent can fix the input and try again. Up to 3 charts per answer.
4. Non-tabular JSON files are queried by the LangChain JSON agent, without charts.
5. The final answer is generated with the configured output prompt and chat context; the charts, if any, are appended
   to the answer as markdown images. If the final LLM call fails, the answer of the agent is returned as it is, with
   the charts. The recent conversation is passed to the agent and the answer is saved in the conversation history, so
   follow-up requests ("now show it as a pie chart") keep their context; in the history the inline charts are replaced
   by a `[chart: title]` placeholder, because the core sends the latest messages to the LLM.

## Uploading Datasets On The Fly

Uploaded datasets are visible in a single conversation (when a chat id is given) or in every conversation of the
agent (shared datasets); a dataset of the conversation wins over a shared one with the same name. When a conversation
sees at least one uploaded dataset, the configured datasource is not used (unless `use_uploaded_datasets` is off): a
shared dataset therefore replaces the configured datasource in every conversation of the agent.

All the datasets visible in a conversation are exposed as one read-only SQLite database: every CSV file becomes a
table named after the file, every SQLite file brings its own tables (the files are processed in alphabetical order and
a clashing table name gets a suffix, e.g. `sales_2`). A single SQLite file is used as it is, with its views and
indexes. Uploading a dataset with the name of an existing one replaces it. Every request reads the datasets as they
were when it started, even if they are replaced meanwhile.

CSV files (both uploaded and configured through `host`) are read detecting separator and encoding. In the files
separated by `;`, as exported with European locales, the numbers are read column by column: a column is read with `,`
as decimal mark and `.` as thousands separator when its values use the comma (`2,5` is 2.5, `1.200,5` is 1200.5) or
when all its values with a dot are thousands groups (`1.200` is 1200); otherwise the usual format is used (`1.5` is
1.5). Columns with `dd/mm/yyyy` dates are converted to ISO dates (`yyyy-mm-dd`), so that they can be sorted and
grouped.

### Through the Rabbit Hole

Upload a CSV or SQLite file with the standard upload (`POST /rabbithole/{chat_id}` for a conversation,
`POST /rabbithole/` for the whole agent). The plugin registers the file as a dataset and stores in the Cat's memory a
description of it (tables, columns, first rows) instead of the raw content: with `capture_rabbithole_uploads` on, CSV
files are no longer ingested as text for the RAG. Other text files are ingested as usual, and so are the CSV files that
cannot be read as tables.

- The Rabbit Hole requires the upload permission (`UPLOAD/WRITE`).
- The core recognizes CSV files only when their first 8 KB are valid UTF-8: CSV files with other encodings must be
  uploaded through the plugin API.
- A dataset larger than `max_upload_size_mb` makes the ingestion fail with an explicit error, notified to the user.

### Through the plugin API

All endpoints require the usual authentication and the `X-Agent-ID` header (or `agent_id` query parameter). Pass the
chat id (`X-Chat-ID` header or `chat_id` query parameter) to work on the datasets of a conversation; without the chat
id you work on the shared datasets. The endpoints require the `CHAT` permission (`WRITE` to upload, `READ` to list,
`DELETE` to delete); uploading or deleting a shared dataset also requires `UPLOAD/WRITE`. Note that the default
permissions of a new user include `CHAT`, so every user can upload datasets in their conversations.

- `POST /custom/cat-with-your-data/datasets`: upload a dataset (multipart form, field `file`)
- `GET /custom/cat-with-your-data/datasets`: list the visible datasets, with tables and columns
- `DELETE /custom/cat-with-your-data/datasets/{name}`: delete a dataset

```bash
curl -X POST "http://localhost:1865/custom/cat-with-your-data/datasets?chat_id=my-chat" \
  -H "Authorization: Bearer $TOKEN" -H "X-Agent-ID: my-agent" \
  -F "file=@sales.csv"
```

### Chart endpoint

- `GET /custom/cat-with-your-data/charts/{agent_id}/{chart_id}.png`: returns a chart generated with
  `chart_delivery` set to `url`. This endpoint is not authenticated, because `<img>` tags cannot send credentials:
  the chart id is a random 128-bit token and works as a capability URL. Charts are deleted after
  `chat_datasets_ttl_hours`.

## Usage

1. Install the plugin in your Grinning Cat environment.
2. Configure plugin settings (or start from one file in `settings_examples/`), or just upload a CSV/SQLite file.
3. Ask questions in natural language, for example:
   - "How many products are in the catalog?"
   - "Show me total sales by month as a line chart."
   - "Which item has the highest revenue? Draw a bar chart of the top 10."
   - "Now show it as a pie chart."

## Notes

- For `CSV`, `JSON` and `SQLite`, set `host` to a readable file path.
- For SQL datasources, verify connectivity and credentials from the Cat runtime environment. The chart queries are
  validated as read-only, and the uploaded datasets and the configured CSV and JSON files are opened read-only, but
  the standard `sql_db_query` tool of the SQL agent can run any statement on the configured SQL databases: use a
  database user with read-only privileges. On every SQLite connection of the plugin (configured SQLite file included)
  `ATTACH` is disabled, so the agent cannot open other files of the host.
- The schema of the configured SQL datasources is cached for 10 minutes: changes of the tables are seen after that.
- Uploaded datasets and URL charts are stored on the local disk (`<cat data folder>/cat_with_your_data`): with
  multiple replicas, this folder must be on a shared volume.
- Charts are drawn by the agent within its usual reasoning steps: no extra LLM call is needed.
- Unlike pandas-ai, the LLM does not write Python code: it writes SQL and a chart specification, and the chart is
  rendered by the plugin. This is safer and passes the Grinning Cat plugin security scan (which forbids `exec`/`eval`),
  but only the supported chart types can be drawn.
- If needed, tune `input_prompt` to inject query hints and business rules.

## Tests

The tests live in `tests/`: unit tests with 100% branch coverage and stateful, property-based tests (with fault
injection) of the invariants listed in `tests/test_invariants.py`. Run them from the root of grinning-cat-core, in
its virtual environment with the plugin dependencies and the `test` dependency group installed:

```bash
python -m unittest discover -s cat/plugins/cat-with-your-data/tests
# deeper search of the property-based tests
CWYD_EXAMPLES=300 python -m unittest discover -s cat/plugins/cat-with-your-data/tests -p "test_invariants.py"
```

The Cat imports and scans every `.py` file of the plugin: the test modules import only the standard library at import
time and must pass the security scan (`test_plugin_loading.py` checks it).

## Changelog

### 0.2.0

New features:

- Charts drawn by the SQL agent through the `draw_chart` tool (`charts`, `chart_delivery`, `public_base_url`,
  `chart_max_rows`, `thought_max_rows` settings), delivered inline (base64) or through the chart endpoint.
- CSV and SQLite datasets uploaded on the fly, per conversation or for the whole agent, through the Rabbit Hole or the
  plugin endpoints (`use_uploaded_datasets`, `capture_rabbithole_uploads`, `max_upload_size_mb`,
  `chat_datasets_ttl_hours` settings).
- `agent_type` setting, to use native tool calling when the LLM supports it.
- The recent conversation is passed to the agent, so that follow-up questions work.

Changed behaviours:

- The configured CSV file is no longer queried by the pandas agent of `langchain-experimental`, which executes Python
  code written by the LLM: it is loaded in an in-memory SQLite database and queried by the SQL agent.
  `langchain-experimental` is no longer a dependency; `langchain-classic`, `langchain-community`, `matplotlib`,
  `pandas` and `sqlalchemy` are declared explicitly.
- Tabular JSON files are queried by the SQL agent (the JSON agent is still used for the other JSON files).
- In CSV files separated by `;` the numbers in European format (`1.200,5`) are recognized; `dd/mm/yyyy` dates become
  ISO dates.
- The `agent_fast_reply` hook now has priority 0, so that it runs after the core `memory` plugin, whose canned reply
  (when no declarative memory is recalled) used to override the answer of this plugin.
- The answers of the plugin are saved in the conversation history (inline charts replaced by a `[chart: title]`
  placeholder): before, the fast reply skipped the core hook that saves them, so they disappeared from the history.
- With `capture_rabbithole_uploads` on (default), the CSV and SQLite files uploaded through the Rabbit Hole become
  datasets and only their description goes into the memory: CSV files are no longer ingested as text.
- `{chat_history}` in `output_prompt` is now a list of `- who: text` lines (last 10 messages, charts replaced by a
  placeholder) instead of the Python representation of a list.
- Database usernames and passwords are URL-encoded in the connection string, and the password is masked in the logs.

Bug fixes:

- The password was encrypted when saved but never decrypted (`SECRET_SETTINGS` was a string instead of a tuple),
  so the connection string contained the encrypted value.
- `settings.py` used `log` without importing it.
- The `Microsoft SQL Server` connection string used a hard-coded user and the host as password.
- The agent output was returned as the string representation of the whole result dictionary.
- Braces in the thought or in the user message (e.g. JSON data) broke the prompt template of the final answer.
- Settings saved by previous versions are completed with the defaults of the new fields.
- `ds_type` and `port` can be left empty (the examples for CSV and JSON files could not be saved).
