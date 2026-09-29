# Cat With Your Data

A Grinning Cat plugin that lets you query your data in natural language, getting answers **and charts**.

It can reason over SQL databases, SQLite, CSV and JSON files, and over CSV/SQLite files uploaded on the fly.

<img src="./thumb.jpg" width="400" alt="Cat With Your Data thumbnail" />

## What It Does

- Turns user questions into queries on your data, and answers in natural language.
- Draws charts (bar, horizontal bar, line, area, pie, scatter, histogram) when the user asks for them, or
  automatically when a visualization makes the answer clearer. The charts are part of the answer.
- Lets users upload CSV and SQLite files on the fly, per conversation or for the whole agent.
- Keeps the context of the conversation, so follow-up requests work ("now show it as a pie chart").
- Supports custom `input_prompt` and `output_prompt` templates.

Unlike pandas-ai, the LLM never writes Python code: it writes SQL and a chart specification, and the plugin draws the
chart.

## Supported Datasources

- `PostgreSQL`
- `MySQL`
- `Oracle`
- `Microsoft SQL Server`
- `Microsoft SQL Server ODBC`
- `SQLite`
- `CSV`
- `JSON`: tabular JSON files (a list of records, or an object of lists of records) can be charted; the other JSON files
  are queried without charts
- CSV (`.csv`, `.tsv`) and SQLite (`.sqlite`, `.sqlite3`, `.db`, `.db3`) files uploaded on the fly

Reference for SQLAlchemy connection formats: https://docs.sqlalchemy.org/en/20/core/engines.html

## Requirements

- Grinning Cat, with a file manager configured for the agent (to upload datasets and to use CSV, JSON and SQLite files)
- The Python dependencies in `pyproject.toml`, installed with the plugin; depending on your datasource, additional DB
  drivers may be required by SQLAlchemy.

## Configuration

Datasource fields:

- `ds_type`: datasource type (for example `PostgreSQL`, `CSV`, `JSON`); it can be left empty if you only work with
  uploaded datasets
- `host`: DB host; for CSV, JSON and SQLite, the path of the file in the files of the agent (e.g. `sales.csv`, a file
  uploaded in the memory of the agent)
- `port`: DB port (it can be left empty for file datasources)
- `username`: DB username
- `password`: DB password (stored encrypted)
- `database`: DB name
- `input_prompt`: template used before reasoning (`{user_message}` available)
- `output_prompt`: final response template (`{prompt_prefix}`, `{user_message}`, `{thought}`, `{chat_history}` available)

Agent, charts and uploads fields:

- `agent_type`: `auto` (default: native tool calling when the LLM supports it, otherwise ReAct), `react` (works with
  every LLM, more error-prone with small models) or `tool_calling` (needs an LLM with tools)
- `charts`: `disabled`, `on_request` (default: a chart is drawn only when the user asks for it, in any language) or
  `auto` (a chart is drawn also when comparisons, rankings, trends or distributions are easier to read visually)
- `chart_max_rows`: maximum number of rows fetched to draw a chart
- `thought_max_rows`: maximum number of rows of the chart data shown to the agent after drawing a chart
- `use_uploaded_datasets`: when a conversation has uploaded datasets, query them instead of the configured datasource
- `capture_rabbithole_uploads`: register the CSV/SQLite files uploaded through the Rabbit Hole as datasets
- `max_upload_size_mb`: maximum size of an uploaded dataset
- `chat_datasets_ttl_hours`: hours after which the datasets of a conversation not used meanwhile are deleted (`0` =
  never; shared datasets never expire)

Ready-to-use examples are in `settings_examples/`.

**Use a database user with read-only privileges** (e.g. on PostgreSQL `GRANT SELECT` on the tables to query, and
nothing else): the plugin accepts only read-only queries without side effects, but only the privileges of the user
guarantee that nothing can be changed.

## Uploading Datasets On The Fly

Uploaded datasets are visible only to the user who uploaded them, in a single conversation (when a chat id is given),
or in every conversation of the agent (shared datasets). A dataset of the conversation wins over a shared one with the
same name, and uploading a dataset with the name of an existing one replaces it. When a conversation sees at least one
uploaded dataset, the configured datasource is not used (unless `use_uploaded_datasets` is off).

All the datasets visible in a conversation are queried together: every CSV file becomes a table named after the file,
every SQLite file brings its own tables (a clashing table name gets a suffix, e.g. `sales_2`).

The datasets of a conversation are deleted with the conversation; the shared ones with the agent.

CSV files are read detecting separator and encoding; in the files separated by `;` (European locales) numbers such as
`1.200,5` are recognized, and `dd/mm/yyyy` dates become ISO dates.

### Through the Rabbit Hole

Upload a CSV or SQLite file with the standard upload (`POST /rabbithole/{chat_id}` for a conversation,
`POST /rabbithole/` for the whole agent). The plugin registers the file as a dataset and stores in the memory of the Cat a
description of it (tables, columns, first rows) instead of the raw content. Other files are ingested as usual.

- The Rabbit Hole requires the upload permission (`UPLOAD/WRITE`).
- The core recognizes CSV files only when their first 8 KB are valid UTF-8: CSV files with other encodings must be
  uploaded through the plugin API.

### Through the plugin API

All endpoints require the usual authentication and the `X-Agent-ID` header (or `agent_id` query parameter). Pass the
chat id (`X-Chat-ID` header or `chat_id` query parameter) to work on the datasets of a conversation; without the chat
id you work on the shared datasets. The endpoints require the `CHAT` permission (`WRITE` to upload, `READ` to list,
`DELETE` to delete); uploading or deleting a shared dataset also requires `UPLOAD/WRITE`.

- `POST /custom/cat-with-your-data/datasets`: upload a dataset (multipart form, field `file`)
- `GET /custom/cat-with-your-data/datasets`: list the visible datasets, with tables and columns
- `DELETE /custom/cat-with-your-data/datasets/{name}`: delete a dataset

```bash
curl -X POST "http://localhost:1865/custom/cat-with-your-data/datasets?chat_id=my-chat" \
  -H "Authorization: Bearer $TOKEN" -H "X-Agent-ID: my-agent" \
  -F "file=@sales.csv"
```

## Usage

1. Install the plugin in your Grinning Cat environment.
2. Configure plugin settings (or start from one file in `settings_examples/`), or just upload a CSV/SQLite file.
3. Ask questions in natural language, for example:
   - "How many products are in the catalog?"
   - "Show me total sales by month as a line chart."
   - "Which item has the highest revenue? Draw a bar chart of the top 10."
   - "Now show it as a pie chart."

If needed, tune `input_prompt` to inject query hints and business rules.

## Tests

From the root of grinning-cat-core, in its virtual environment:

```bash
python -m unittest discover -s cat/plugins/cat-with-your-data/tests
```

## Changelog

### 0.2.0

New features:

- Charts in the answers (`charts`, `chart_max_rows`, `thought_max_rows` settings).
- CSV and SQLite datasets uploaded on the fly, per conversation or for the whole agent, through the Rabbit Hole or the
  plugin endpoints (`use_uploaded_datasets`, `capture_rabbithole_uploads`, `max_upload_size_mb`,
  `chat_datasets_ttl_hours` settings).
- `agent_type` setting, to use native tool calling when the LLM supports it.
- Follow-up questions keep the context of the conversation.

Changed behaviours:

- The configured CSV file is no longer queried by the LLM writing Python code: it is queried with SQL.
- Tabular JSON files can be charted.
- In CSV files separated by `;` the European number format is recognized; `dd/mm/yyyy` dates become ISO dates.
- The answers of the plugin are saved in the conversation history (charts replaced by a `[chart: title]` placeholder).
- The configured CSV, JSON and SQLite files are read from the files of the agent (`host`: their path).
- The datasets of a conversation belong to the user who uploaded them.
- With `capture_rabbithole_uploads` on (default), the CSV and SQLite files uploaded through the Rabbit Hole become
  datasets and only their description goes into the memory.
- `{chat_history}` in `output_prompt` is now a list of `- who: text` lines.
- The agent can only read the configured databases: only read-only queries without side effects are accepted.

Bug fixes:

- The password was encrypted when saved but never decrypted.
- The `Microsoft SQL Server` connection string used a hard-coded user and the host as password.
- The agent output was returned as the string representation of the whole result dictionary.
- Braces in the thought or in the user message (e.g. JSON data) broke the final answer.
- Settings saved by previous versions are completed with the defaults of the new fields.
- `ds_type` and `port` can be left empty.
