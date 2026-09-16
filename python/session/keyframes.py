"""Session-owned preset metadata, independent of physics and panel state."""

from mojive.adapters.base import KeyframeCatalog, KeyframeInfo


class ModelKeyframeCatalog:
    """Index metadata once per content change and retain stable per-model views."""

    def __init__(self) -> None:
        self.items: list[KeyframeInfo] = []
        self.revision = 0
        self._signature: tuple[tuple[int, str, float, int], ...] = ()
        self._models: dict[int, tuple[KeyframeInfo, ...]] = {}
        self._slots: dict[int, int] = {}

    def refresh(self, provider: KeyframeCatalog | None) -> None:
        items = list(provider.keyframes()) if provider is not None else []
        signature = tuple((item.keyframe_id, item.name, item.time, item.model_id) for item in items)
        if signature == self._signature:
            return
        models: dict[int, list[KeyframeInfo]] = {}
        for item in items:
            models.setdefault(item.model_id, []).append(item)
        self.items = items
        self._signature = signature
        self._slots = {item.keyframe_id: slot for slot, item in enumerate(items)}
        self._models = {
            model: tuple(sorted(values, key=lambda item: (item.time, item.keyframe_id)))
            for model, values in models.items()
        }
        self.revision += 1

    def for_model(self, model_id: int) -> tuple[KeyframeInfo, ...]:
        """Return a stable sorted view until preset metadata changes."""
        return self._models.get(model_id, ())

    def slot(self, keyframe_id: int) -> int:
        """Return the metadata slot for an identity, or -1 when it has disappeared."""
        return self._slots.get(keyframe_id, -1)
