"""HTTP endpoints of the plugin.

Datasets (CSV / SQLite) can be uploaded on the fly:

- with ``chat_id`` (query parameter or ``X-Chat-ID`` header) the dataset is visible only to the user, in that
  conversation, and the CHAT/WRITE permission is enough;
- without ``chat_id`` the dataset is shared by every conversation of the agent and the UPLOAD/WRITE permission is
  required.
"""
import asyncio
from typing import Dict, List

from fastapi import UploadFile
from pydantic import BaseModel

from cat import AuthorizedInfo, AuthPermission, AuthResource, check_permissions, endpoint, log
from cat.exceptions import CustomForbiddenException, CustomNotFoundException, CustomValidationException
from cat.routes.routes_utils import has_write_permission

# sibling modules are looked up at call time
from . import datasets

TAGS = ["Cat With Your Data"]
BASE_PATH = "/cat-with-your-data"


class DatasetResponse(BaseModel):
    name: str
    kind: str
    scope: str
    size: int
    uploaded_at: float
    tables: Dict[str, List[str]]


class DatasetListResponse(BaseModel):
    datasets: List[DatasetResponse]


class DatasetDeleteResponse(BaseModel):
    deleted: bool


def _store(info: AuthorizedInfo) -> datasets.DatasetStore:
    if info.cheshire_cat is None:
        raise CustomValidationException("The agent is required (X-Agent-ID header or agent_id query parameter)")
    chat_id = info.stray_cat.id if info.stray_cat else None
    return datasets.DatasetStore(info.cheshire_cat.file_manager, info.cheshire_cat.agent_key, chat_id, info.user.id)


def _check_shared_permission(info: AuthorizedInfo) -> None:
    if info.stray_cat is None and not has_write_permission(info.user.permissions, AuthResource.UPLOAD):
        raise CustomForbiddenException("The UPLOAD/WRITE permission is required to manage shared datasets")


async def _settings(info: AuthorizedInfo) -> Dict:
    try:
        return await info.cheshire_cat.mad_hatter.get_plugin().load_settings(info.cheshire_cat.agent_key)
    except Exception as e:
        log.warning(f"[cat-with-your-data] cannot load the settings: {e}")
        return {}


@endpoint.post(f"{BASE_PATH}/datasets", response_model=DatasetResponse, tags=TAGS)
async def upload_dataset(
    file: UploadFile,
    info: AuthorizedInfo = check_permissions(AuthResource.CHAT, AuthPermission.WRITE),
) -> DatasetResponse:
    """Upload a CSV (.csv, .tsv) or SQLite file and make it queryable, with charts, in natural language."""
    store = _store(info)
    _check_shared_permission(info)

    settings = await _settings(info)
    max_bytes = int(settings.get("max_upload_size_mb") or 100) * 1024 * 1024
    if file.size is not None and file.size > max_bytes:
        raise CustomValidationException(f"The file exceeds the maximum allowed size of {max_bytes // (1024 * 1024)} MB")

    content = await file.read()
    try:
        dataset = await asyncio.to_thread(
            store.add, file.filename or "dataset", content, info.stray_cat is None, max_bytes
        )
    except datasets.DatasetError as e:
        raise CustomValidationException(str(e)) from e

    await asyncio.to_thread(store.cleanup_expired, float(settings.get("chat_datasets_ttl_hours") or 0))
    return DatasetResponse(**dataset.to_dict())


@endpoint.get(f"{BASE_PATH}/datasets", response_model=DatasetListResponse, tags=TAGS)
async def list_datasets(
    info: AuthorizedInfo = check_permissions(AuthResource.CHAT, AuthPermission.READ),
) -> DatasetListResponse:
    """List the datasets visible in the conversation (``chat_id``) or the shared ones (no ``chat_id``)."""
    items = await asyncio.to_thread(_store(info).list_datasets, True)
    return DatasetListResponse(datasets=[DatasetResponse(**d.to_dict()) for d in items])


@endpoint.delete(f"{BASE_PATH}/datasets/{{name}}", response_model=DatasetDeleteResponse, tags=TAGS)
async def delete_dataset(
    name: str,
    info: AuthorizedInfo = check_permissions(AuthResource.CHAT, AuthPermission.DELETE),
) -> DatasetDeleteResponse:
    """Delete a dataset of the conversation (``chat_id``) or a shared one (no ``chat_id``)."""
    store = _store(info)
    _check_shared_permission(info)

    try:
        deleted = await asyncio.to_thread(store.remove, name, info.stray_cat is None)
    except datasets.DatasetError as e:
        raise CustomValidationException(str(e)) from e
    if not deleted:
        raise CustomNotFoundException("Dataset not found")
    return DatasetDeleteResponse(deleted=True)
