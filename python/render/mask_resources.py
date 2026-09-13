"""Bounded sampled-image residency with references pinned until frame submission."""

from collections import OrderedDict


class MaskCache:
    def __init__(self, create, release, *, max_bytes=16 * 1024 * 1024, max_entries=256):
        if any(
            not isinstance(v, int) or isinstance(v, bool) or v < 1 for v in (max_bytes, max_entries)
        ):
            raise ValueError("Mask cache budgets must be positive integers")
        self.create, self.release = create, release
        self.max_bytes, self.max_entries = max_bytes, max_entries
        self.entries = OrderedDict()
        self.pinned = set()
        self.resident_bytes = self.upload_bytes = self.uploads = self.evictions = 0

    def begin_frame(self):
        self.pinned.clear()

    def get(self, mask):
        if mask in self.entries:
            self.entries.move_to_end(mask)
            self.pinned.add(mask)
            return self.entries[mask]
        if mask.byte_size > self.max_bytes:
            raise ValueError("Coverage mask exceeds resident mask budget")
        while (
            self.resident_bytes + mask.byte_size > self.max_bytes
            or len(self.entries) >= self.max_entries
        ):
            victim = next((key for key in self.entries if key not in self.pinned), None)
            if victim is None:
                raise ValueError("Mask budget exhausted by pending frame resources")
            self.release(self.entries.pop(victim))
            self.resident_bytes -= victim.byte_size
            self.evictions += 1
        resource = self.create(mask)
        self.entries[mask] = resource
        self.pinned.add(mask)
        self.resident_bytes += mask.byte_size
        self.upload_bytes += mask.byte_size
        self.uploads += 1
        return resource

    def stats(self):
        return {
            "mask_uploads": self.uploads,
            "mask_upload_bytes": self.upload_bytes,
            "mask_resident_bytes": self.resident_bytes,
            "mask_entries": len(self.entries),
            "mask_evictions": self.evictions,
        }

    def close(self):
        for resource in self.entries.values():
            self.release(resource)
        self.entries.clear()
        self.pinned.clear()
        self.resident_bytes = 0
