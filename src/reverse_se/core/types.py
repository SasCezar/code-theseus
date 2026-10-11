from pandas import DataFrame
from pydantic import BaseModel, ConfigDict


class Repository(BaseModel):
    name: str
    url: str | None = None
    description: str | None = None
    language: str | None = None
    archived: bool | None = None
    pushed_at: str | None = None
    stargazers_count: int | None = None


class RepoContext(BaseModel):
    name: str
    url: str
    language: str
    nodes: DataFrame | None = None
    edges: DataFrame | None = None
    package_files: DataFrame | None = None
    architecture_doc: str | None = None
    requirments_doc: str | None = None
    description_doc: str | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)
