"""Draft and apply a bounded subset without doing work while typing."""

from imgui_bundle import imgui

from mojive import commands as cmd


class WorldSelectionControl:
    """Shared world selection editor; authoritative state belongs to Session."""

    def __init__(self):
        self._token = None
        self._start = 0
        self._count = 1
        self._ids = ""
        self._explicit = False
        self._error = ""

    def draw(self, ctx):
        """Show range/count or explicit original IDs, committing only on Apply."""
        info = ctx.session.world_selection
        if info is None:
            return
        sync = ctx.session.rollout_sync_info
        token = (ctx.session.document_id, info)
        if token != self._token:
            self._token = token
            self._start, self._count = info.world_ids[0], len(info.world_ids)
            self._ids = ", ".join(map(str, info.world_ids))
            self._explicit = info.world_ids != tuple(range(self._start, self._start + self._count))
            self._error = ""
        t = ctx.tr
        imgui.separator()
        imgui.text(f"{t('Worlds')}: {len(info.world_ids)} / {info.total_worlds}")
        imgui.text_disabled(f"{t('Preview limit')}: {info.max_worlds}")
        imgui.begin_disabled(sync is not None and sync.pending)
        _, self._explicit = imgui.checkbox(t("Choose world IDs"), self._explicit)
        if self._explicit:
            imgui.set_next_item_width(-1)
            _, self._ids = imgui.input_text_with_hint("##world_ids", "0, 7, 42", self._ids)
        else:
            imgui.text_disabled(t("Start world ID"))
            imgui.set_next_item_width(-1)
            _, self._start = imgui.input_int("##world_start", self._start)
            imgui.text_disabled(t("World count"))
            imgui.set_next_item_width(-1)
            _, self._count = imgui.input_int("##world_count", self._count)
        applied = imgui.button(t("Sync worlds" if sync is not None else "Apply worlds"))
        imgui.end_disabled()
        if applied:
            try:
                if self._explicit:
                    ids = tuple(int(i.strip()) for i in self._ids.split(","))
                else:
                    if not 1 <= self._count <= info.max_worlds:
                        raise ValueError(t("Count exceeds preview limit"))
                    ids = tuple(range(self._start, self._start + self._count))
                result = ctx.submit(
                    cmd.SyncRollout(ids) if sync is not None else cmd.SetWorldSelection(ids)
                )
                self._error = "" if result.ok else result.message
            except ValueError:
                self._error = t("Enter valid world IDs within the preview limit")
        if self._error:
            imgui.text_wrapped(self._error)
        if sync is not None:
            if sync.pending:
                imgui.text(t("Downloading rollout..."))
            elif sync.error:
                imgui.text_wrapped(sync.error)
