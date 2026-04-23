from __future__ import annotations

import uuid
from typing import Dict, List
from .schemas import ProjectState, ChatMessage


class SessionStore:
    def __init__(self) -> None:
        self.projects: Dict[str, ProjectState] = {}
        self.messages: Dict[str, List[ChatMessage]] = {}

    def has_project(self, project_id: str) -> bool:
        return project_id in self.projects

    def create_project(self, state: ProjectState) -> str:
        project_id = str(uuid.uuid4())
        self.projects[project_id] = state
        self.messages[project_id] = []
        return project_id

    def get_state(self, project_id: str) -> ProjectState:
        return self.projects[project_id]

    def save_state(self, project_id: str, state: ProjectState) -> None:
        self.projects[project_id] = state

    def append_message(self, project_id: str, role: str, content: str) -> None:
        self.messages.setdefault(project_id, []).append(ChatMessage(role=role, content=content))

    def get_messages(self, project_id: str) -> List[ChatMessage]:
        return self.messages.get(project_id, [])


STORE = SessionStore()

_impl: SessionStore | None = None


def get_store() -> SessionStore:
    """Memory store by default; set DATABASE_URL for Postgres or SQLite persistence."""
    global _impl
    if _impl is None:
        import os

        dsn = os.getenv("DATABASE_URL", "").strip()
        if dsn:
            from .sql_store import SqlStore

            _impl = SqlStore(dsn)
        else:
            _impl = SessionStore()
    return _impl
