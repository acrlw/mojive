"""Flush atomically published draft snapshots without blocking native frames."""

import os
from concurrent.futures import ThreadPoolExecutor


def _flush(stream):
    with stream:
        os.fsync(stream.fileno())


class DraftSync:
    """Own one in-flight snapshot and coalesce subsequent durability requests.

    The editor publishes each revision synchronously by atomic replacement.
    Opening the snapshot on the caller thread pins that inode until fsync ends;
    the worker never locks, reads or mutates editor state or the current path.
    """

    def __init__(self, path):
        self.path = path
        self.dirty = False
        self.error = None
        self._future = None
        self._executor = None

    def _complete(self):
        future, self._future = self._future, None
        try:
            future.result()
        except OSError as error:
            self.error = error
            self.dirty = True
            raise
        self.error = None

    def poll(self):
        if self._future is not None:
            if not self._future.done():
                return
            self._complete()
        if self.dirty:
            stream = self.path.open("rb")
            if self._executor is None:
                self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="design-sync")
            try:
                self._future = self._executor.submit(_flush, stream)
            except Exception:
                stream.close()
                raise
            self.dirty = False

    def flush(self):
        """Confirm all requested revisions at explicit save and shutdown boundaries."""
        if self._future is not None:
            self._complete()
        if self.dirty:
            _flush(self.path.open("rb"))
            self.dirty = False
            self.error = None

    def close(self):
        try:
            self.flush()
        finally:
            if self._executor is not None:
                self._executor.shutdown(wait=True)
                self._executor = None
