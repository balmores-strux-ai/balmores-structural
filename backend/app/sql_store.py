from __future__ import annotations

import uuid
from pathlib import Path
from typing import List

from sqlalchemy import ForeignKey, Integer, String, Text, create_engine, desc, select
from sqlalchemy.engine import make_url
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship

from .schemas import ChatMessage, ProjectState


class Base(DeclarativeBase):
    pass


class ProjectRow(Base):
    __tablename__ = "balmores_projects"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    state_json: Mapped[str] = mapped_column(Text)

    messages: Mapped[List["MessageRow"]] = relationship(
        back_populates="project",
        cascade="all, delete-orphan",
        order_by="MessageRow.seq",
    )


class MessageRow(Base):
    __tablename__ = "balmores_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    project_id: Mapped[str] = mapped_column(String(36), ForeignKey("balmores_projects.id", ondelete="CASCADE"), index=True)
    seq: Mapped[int] = mapped_column(Integer, index=True)
    role: Mapped[str] = mapped_column(String(32))
    content: Mapped[str] = mapped_column(Text)

    project: Mapped["ProjectRow"] = relationship(back_populates="messages")


def _ensure_sqlite_parent_dir(url: str) -> str:
    try:
        u = make_url(url)
        if u.drivername.startswith("sqlite") and u.database and u.database != ":memory:":
            Path(u.database).parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return url


class SqlStore:
    """Persistent project + message store (Postgres or SQLite)."""

    def __init__(self, database_url: str) -> None:
        url = database_url
        if url.startswith("postgres://"):
            url = "postgresql://" + url[len("postgres://") :]
        url = _ensure_sqlite_parent_dir(url)
        self._engine = create_engine(url, pool_pre_ping=True)
        Base.metadata.create_all(self._engine)

    def has_project(self, project_id: str) -> bool:
        with Session(self._engine) as session:
            row = session.get(ProjectRow, project_id)
            return row is not None

    def create_project(self, state: ProjectState) -> str:
        project_id = str(uuid.uuid4())
        with Session(self._engine) as session:
            session.add(ProjectRow(id=project_id, state_json=state.model_dump_json()))
            session.commit()
        return project_id

    def get_state(self, project_id: str) -> ProjectState:
        with Session(self._engine) as session:
            row = session.get(ProjectRow, project_id)
            if row is None:
                raise KeyError(project_id)
            return ProjectState.model_validate_json(row.state_json)

    def save_state(self, project_id: str, state: ProjectState) -> None:
        with Session(self._engine) as session:
            row = session.get(ProjectRow, project_id)
            if row is None:
                raise KeyError(project_id)
            row.state_json = state.model_dump_json()
            session.commit()

    def append_message(self, project_id: str, role: str, content: str) -> None:
        with Session(self._engine) as session:
            q = (
                select(MessageRow.seq)
                .where(MessageRow.project_id == project_id)
                .order_by(desc(MessageRow.seq))
                .limit(1)
            )
            last = session.execute(q).scalar_one_or_none()
            nxt = (last or 0) + 1
            session.add(MessageRow(project_id=project_id, seq=nxt, role=role, content=content))
            session.commit()

    def get_messages(self, project_id: str) -> List[ChatMessage]:
        with Session(self._engine) as session:
            q = select(MessageRow).where(MessageRow.project_id == project_id).order_by(MessageRow.seq.asc())
            rows = session.scalars(q).all()
            return [ChatMessage(role=r.role, content=r.content) for r in rows]
