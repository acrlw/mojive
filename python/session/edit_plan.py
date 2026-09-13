"""Prepared command plans and results shared by UI and control transactions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from mojive.adapters.base import NodeType
from mojive.commands import Command, CommandResult

if TYPE_CHECKING:
    from .core import Session

NodeIdentity = tuple[int, NodeType, str]
Commands = tuple[Command, ...]


@dataclass(frozen=True)
class EditVersion:
    """Document and adapter structure addressed by a prepared command plan."""

    document_id: str
    document_revision: int
    structure_revision: int

    @classmethod
    def capture(cls, session: Session) -> EditVersion:
        return cls(
            session.document_id, session.document_revision, session.adapter.structure_revision
        )


@dataclass(frozen=True)
class ModelEditPlan:
    """Install creations before binding dependent edits to their actual identities.

    UI drafts defer renames until existing node references have been consumed.
    Ordered control requests disable that scheduling and retain their given order.
    """

    direct: Commands
    creations: Commands = ()
    bind: Callable[[Session, tuple[NodeIdentity, ...]], Commands] | None = None
    rebind: Callable[[Session, Commands], Commands] | None = None
    renames_last: bool = True
    version: EditVersion | None = None


@dataclass(frozen=True)
class EditBatchResult:
    """Final transaction outcome and ordered command results.

    ``failed_index`` addresses execution order, matching input order for apply_edits.
    A missing index denotes preparation, group finalization or history failure.
    ``rollback_error`` retains recovery failure separately from the original error.
    Results from a failed batch are diagnostic and do not represent committed edits.
    """

    result: CommandResult
    results: tuple[CommandResult, ...] = ()
    failed_index: int | None = None
    rollback_error: str = ""
