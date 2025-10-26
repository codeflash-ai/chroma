import threading
import uuid
from typing import Any, Callable
from chromadb.types import Segment
from overrides import override
from typing import Dict, Optional
from abc import ABC, abstractmethod


class SegmentCache(ABC):
    @abstractmethod
    def get(self, key: uuid.UUID) -> Optional[Segment]:
        pass

    @abstractmethod
    def pop(self, key: uuid.UUID) -> Optional[Segment]:
        pass

    @abstractmethod
    def set(self, key: uuid.UUID, value: Segment) -> None:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass


class BasicCache(SegmentCache):
    def __init__(self):
        self.cache: Dict[uuid.UUID, Segment] = {}
        self.lock = threading.RLock()

    @override
    def get(self, key: uuid.UUID) -> Optional[Segment]:
        with self.lock:
            return self.cache.get(key)

    @override
    def pop(self, key: uuid.UUID) -> Optional[Segment]:
        with self.lock:
            return self.cache.pop(key, None)

    @override
    def set(self, key: uuid.UUID, value: Segment) -> None:
        with self.lock:
            self.cache[key] = value

    @override
    def reset(self) -> None:
        with self.lock:
            self.cache = {}


class SegmentLRUCache(BasicCache):
    """A simple LRU cache implementation that handles objects with dynamic sizes.
    The size of each object is determined by a user-provided size function."""

    def __init__(
        self,
        capacity: int,
        size_func: Callable[[uuid.UUID], int],
        callback: Optional[Callable[[uuid.UUID, Segment], Any]] = None,
    ):
        self.capacity = capacity
        self.size_func = size_func
        self.cache: Dict[uuid.UUID, Segment] = {}
        # Use a dict to maintain LRU order and O(1) upsert/remove
        self.history: Dict[uuid.UUID, None] = {}
        self.callback = callback
        self.lock = threading.RLock()
        # Maintain running sizes for O(1) queries and updates
        self.key_sizes: Dict[uuid.UUID, int] = {}
        self.total_size: int = 0

    def _upsert_key(self, key: uuid.UUID):
        # Remove and re-insert to move to end for LRU order (preserved order in Python 3.7+ dicts)
        if key in self.history:
            self.history.pop(key)
        self.history[key] = None

    @override
    def get(self, key: uuid.UUID) -> Optional[Segment]:
        with self.lock:
            self._upsert_key(key)
            if key in self.cache:
                return self.cache[key]
            else:
                return None

    @override
    def pop(self, key: uuid.UUID) -> Optional[Segment]:
        with self.lock:
            if key in self.history:
                self.history.remove(key)
            return self.cache.pop(key, None)

    @override
    def set(self, key: uuid.UUID, value: Segment) -> None:
        with self.lock:
            if key in self.cache:
                return
            item_size = self.size_func(key)

            # Evict items if capacity is exceeded (O(1) per item)
            while self.total_size + item_size > self.capacity and self.history:
                # FIFO order for oldest (LRU) key: get the first key
                key_delete = next(iter(self.history))
                if key_delete in self.cache:
                    self.callback(key_delete, self.cache[key_delete])
                    del self.cache[key_delete]
                    size_deleted = self.key_sizes.pop(key_delete)
                    self.total_size -= size_deleted
                self.history.pop(key_delete)

            self.cache[key] = value
            self.key_sizes[key] = item_size
            self.total_size += item_size
            self._upsert_key(key)

    @override
    def reset(self):
        with self.lock:
            self.cache = {}
            self.history = []
