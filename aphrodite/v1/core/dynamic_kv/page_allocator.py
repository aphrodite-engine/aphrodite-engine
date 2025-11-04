import threading
from collections import deque
from typing import cast

import torch

from .locks import ConditionLike, LockLike, NoOpCondition, NoOpLock
from .mem_info_tracker import MemInfoTracker
from .tp_ipc_util import broadcast_map_to_kv_tensors, broadcast_unmap_from_kv_tensors
from .utils import (
    CONTIGUOUS_LAYOUT,
    GPU_UTILIZATION,
    MAX_RESERVED_PAGES,
    MIN_RESERVED_PAGES,
    PAGE_PREALLOC_ENABLED,
    SANITY_CHECK,
)

try:
    from aphrodite.v1.core.dynamic_kv import vmm_ops

    _vmm_ops_available = True
except ImportError:
    _vmm_ops_available = False

from aphrodite.logger import init_logger

logger = init_logger(__name__)

PREALLOC_THREAD_TIMEOUT: float = 2.0


class Page:
    def __init__(self, page_id: int, page_size: int):
        self.page_id = page_id
        self.page_size = page_size

        self.start_block: int | None = None
        self.end_block: int | None = None
        self.num_kv_blocks: int | None = None
        self.free_list: list[int] = []

    def _require_init(self) -> None:
        """Raise AssertionError if the page has not been initialised."""
        assert self.start_block is not None, "Page not initialised"
        assert self.end_block is not None, "Page not initialised"
        assert self.num_kv_blocks is not None, "Page not initialised"

    def init(self, block_mem_size: int) -> None:
        self.start_block, self.end_block = self.get_block_range(self.page_id, self.page_size, block_mem_size)

        self.num_kv_blocks = self.end_block - self.start_block
        self.free_list = list(range(self.start_block, self.end_block))

    def alloc(self, num_blocks: int = 1) -> list[int]:
        self._require_init()
        if self.full():
            raise ValueError(f"Page {self.page_id} is already full")
        block_ids = self.free_list[:num_blocks]
        self.free_list = self.free_list[num_blocks:]
        return block_ids

    def free(self, block_id: int) -> None:
        self._require_init()
        if SANITY_CHECK:
            self._sanity_check(block_id)
        self.free_list.append(block_id)

    def free_batch(self, block_ids: list[int]) -> None:
        self._require_init()
        if SANITY_CHECK:
            for block_id in block_ids:
                self._sanity_check(block_id)
        self.free_list.extend(block_ids)

    def empty(self) -> bool:
        self._require_init()
        return len(self.free_list) == self.num_kv_blocks

    def full(self) -> bool:
        self._require_init()
        return not self.free_list

    def num_free_blocks(self) -> int:
        self._require_init()
        return len(self.free_list)

    def get_free_blocks(self) -> list[int]:
        self._require_init()
        return self.free_list

    def _has_block(self, block_id: int) -> bool:
        self._require_init()
        return block_id >= cast(int, self.start_block) and block_id < cast(int, self.end_block)

    def _sanity_check(self, block_id: int) -> None:
        self._require_init()
        if not self._has_block(block_id):
            raise ValueError(f"Page {self.page_id} does not have block {block_id}")
        if block_id in self.free_list:
            raise ValueError(f"Block {block_id} is already free")

    @staticmethod
    def get_block_range(page_id: int, page_size: int, block_mem_size: int) -> tuple[int, int]:
        """Get the block range of a page."""
        start_block = (page_id * page_size + block_mem_size - 1) // block_mem_size
        end_block = ((page_id + 1) * page_size) // block_mem_size
        return start_block, end_block

    @staticmethod
    def get_num_blocks(page_size: int, block_mem_size: int) -> int:
        """Calculate the number of blocks that can fit in a page."""
        return page_size // block_mem_size


class PageAllocator:
    def __init__(
        self,
        num_layers: int,
        mem_size_per_layer: int,
        page_size: int,
        tp_size: int = 1,
        async_sched: bool = False,
        contiguous_layout: bool = CONTIGUOUS_LAYOUT,
        enable_page_prealloc: bool = PAGE_PREALLOC_ENABLED,
    ):
        """
        Args:
            num_layers: Number of layers (for physical memory calculation).
            mem_size_per_layer: Memory size per layer per K/V tensor in bytes.
            page_size: Page size in bytes.
            tp_size: Tensor parallel size.
            async_sched: Whether asynchronous scheduling is enabled.
            contiguous_layout: Whether to use contiguous layout.
            enable_page_prealloc: Whether to enable page preallocation.
        """
        logger.info(
            "Init kvcached KV cache allocator: "
            "num_layers=%d, "
            "mem_size_per_layer=%dMB, "
            "total_mem_size=%dMB, "
            "page_size=%d, "
            "tp_size=%d, "
            "async_sched=%s, "
            "contiguous_layout=%s, "
            "enable_prealloc=%s",
            num_layers,
            mem_size_per_layer // (1024 * 1024),
            2 * num_layers * mem_size_per_layer // (1024 * 1024),
            page_size // (1024 * 1024),
            tp_size,
            async_sched,
            contiguous_layout,
            enable_page_prealloc,
        )

        self.num_layers = num_layers
        self.mem_size_per_layer = mem_size_per_layer
        self.page_size = page_size
        self.tp_size = tp_size
        self.async_sched = async_sched
        self.contiguous_layout = contiguous_layout
        self.gpu_utilization = GPU_UTILIZATION
        self.num_free_pages = mem_size_per_layer // page_size
        self.num_total_pages = mem_size_per_layer // page_size

        self.free_page_list: deque[int] = deque(range(self.num_free_pages))

        self.min_reserved_pages: int = MIN_RESERVED_PAGES
        self.max_reserved_pages: int = MAX_RESERVED_PAGES
        self.reserved_page_list: deque[int] = deque()

        self.reclaimed_page_list: deque[int] = deque()

        self.mem_info_tracker = MemInfoTracker(self.mem_size_per_layer * num_layers * 2)

        self.enable_page_prealloc: bool = enable_page_prealloc

        self._lock: LockLike
        self._cond: ConditionLike

        if self.enable_page_prealloc:
            self._lock = threading.RLock()
            self._cond = threading.Condition(self._lock)
        else:
            self._lock = NoOpLock()
            self._cond = NoOpCondition(self._lock)
        self.prealloc_running: bool = False
        self.prealloc_needed: bool = False
        self.prealloc_thd: threading.Thread | None = None

    def __del__(self):
        try:
            if self.enable_page_prealloc and self.prealloc_thd is not None:
                self._stop_prealloc_thread(timeout=PREALLOC_THREAD_TIMEOUT)
        except Exception:
            pass

    def start_prealloc_thread(self):
        if self.enable_page_prealloc:
            self._lock = threading.RLock()
            self._cond = threading.Condition(self._lock)
            self._start_prealloc_thread()

    def alloc_page(self) -> Page:
        with self._lock:
            page_id: int | None = None

            while page_id is None:
                if self.reserved_page_list:
                    page_id = self.reserved_page_list.popleft()
                    self.num_free_pages -= 1

                    if len(self.reserved_page_list) < self.min_reserved_pages:
                        self.prealloc_needed = True
                        self._cond.notify_all()

                    self._update_memory_usage()
                    return Page(page_id, self.page_size)

                if self.free_page_list:
                    page_id = self.free_page_list.popleft()
                    self.num_free_pages -= 1
                    break

                if self.num_free_pages <= 0:
                    raise ValueError("No free pages left")

                if not self.enable_page_prealloc:
                    raise RuntimeError("Inconsistent page allocator state: no free pages available to allocate")

                self._cond.wait()

        assert page_id is not None

        try:
            self._map_pages([page_id])
        except Exception as e:
            with self._lock:
                self.free_page_list.appendleft(page_id)
                self.num_free_pages += 1
                self._cond.notify_all()
            raise RuntimeError(f"Failed to map page {page_id}: {e}") from e

        if self.enable_page_prealloc:
            self._trigger_preallocation()

        self._update_memory_usage()
        return Page(page_id, self.page_size)

    def free_page(self, page_id: int) -> None:
        with self._lock:
            if SANITY_CHECK and (page_id in self.free_page_list or page_id in self.reserved_page_list):
                raise ValueError(f"Page {page_id} is already free or reserved")

            self.num_free_pages += 1
            if len(self.reserved_page_list) < self.max_reserved_pages:
                self.reserved_page_list.append(page_id)
                self._update_memory_usage()
                self._cond.notify_all()
                return

        self._unmap_pages([page_id])
        with self._lock:
            self.free_page_list.append(page_id)
            self._update_memory_usage()
            self._cond.notify_all()

    def free_pages(self, page_ids: list[int]) -> None:
        with self._lock:
            if SANITY_CHECK:
                for page_id in page_ids:
                    if page_id in self.free_page_list or page_id in self.reserved_page_list:
                        raise ValueError(f"Page {page_id} is already free or reserved")

            self.num_free_pages += len(page_ids)
            num_to_reserve = self.max_reserved_pages - len(self.reserved_page_list)
            if num_to_reserve > 0:
                self.reserved_page_list.extend(page_ids[:num_to_reserve])
                self._cond.notify_all()
                page_ids = page_ids[num_to_reserve:]

        if len(page_ids) == 0:
            self._update_memory_usage()
            return

        self._unmap_pages(page_ids)
        with self._lock:
            self.free_page_list.extend(page_ids)
            self._update_memory_usage()
            self._cond.notify_all()

    def resize(self, new_mem_size: int) -> bool:
        new_num_pages = new_mem_size // self.page_size
        with self._lock:
            if new_num_pages < self.get_num_inuse_pages():
                return False
            if new_num_pages == self.num_total_pages:
                return True
            elif new_num_pages > self.num_total_pages:
                num_to_expand = new_num_pages - self.num_total_pages

                num_to_reuse = min(len(self.reclaimed_page_list), num_to_expand)
                if num_to_reuse > 0:
                    for _ in range(num_to_reuse):
                        self.free_page_list.append(self.reclaimed_page_list.popleft())
                    num_to_expand -= num_to_reuse
                    self.num_free_pages += num_to_reuse

                if num_to_expand > 0:
                    new_page_ids = list(range(self.num_total_pages, self.num_total_pages + num_to_expand))
                    self.free_page_list.extend(new_page_ids)
                    self.num_free_pages += num_to_expand
                self.num_total_pages = new_num_pages
                self._update_memory_usage()
            else:
                num_to_reclaim = self.num_total_pages - new_num_pages

                if len(self.free_page_list) < num_to_reclaim:
                    reserved_count = len(self.reserved_page_list)
                    if reserved_count > 0:
                        pages_to_unmap = list(self.reserved_page_list)
                        self.reserved_page_list.clear()
                        try:
                            self._lock.release()
                            self._unmap_pages(pages_to_unmap)
                        finally:
                            self._lock.acquire()
                        self.free_page_list.extend(pages_to_unmap)
                        self._update_memory_usage()

                if len(self.free_page_list) < num_to_reclaim:
                    return False

                for _ in range(num_to_reclaim):
                    self.reclaimed_page_list.append(self.free_page_list.pop())
                self.num_free_pages -= num_to_reclaim
                self.num_total_pages = new_num_pages
        return True

    def trim(self) -> None:
        with self._lock:
            pages_to_unmap = list(self.reserved_page_list)
            self.reserved_page_list.clear()

            if not pages_to_unmap:
                self._update_memory_usage()
                return

            try:
                self._lock.release()
                self._unmap_pages(pages_to_unmap)
            finally:
                self._lock.acquire()

            self.free_page_list.extend(pages_to_unmap)
            self._update_memory_usage()

    def get_num_free_pages(self) -> int:
        return self.num_free_pages

    def get_num_inuse_pages(self) -> int:
        return self.num_total_pages - self.num_free_pages

    def get_num_total_pages(self) -> int:
        return self.num_total_pages

    def get_num_reserved_pages(self) -> int:
        with self._lock:
            return len(self.reserved_page_list)

    def get_avail_physical_pages(self) -> int:
        avail_phy_mem_size, total_phy_mem_size = torch.cuda.mem_get_info()
        headroom = int(total_phy_mem_size * (1 - self.gpu_utilization))
        avail_phy_mem_size = max(avail_phy_mem_size - headroom, 0)

        avail_phy_pages = avail_phy_mem_size // self.page_size
        avail_pages_per_layer = avail_phy_pages // self.num_layers // 2
        return int(avail_pages_per_layer)

    def get_page_id(self, block_id: int, block_mem_size: int) -> int:
        return block_id * block_mem_size // self.page_size

    def _prealloc_worker(self):
        """Worker thread that preallocates and maps physical pages."""
        while self.prealloc_running:
            with self._lock:
                while not self.prealloc_needed and self.prealloc_running:
                    self._cond.wait()

                if not self.prealloc_running:
                    break

                self.prealloc_needed = False
                current_reserved = len(self.reserved_page_list)
                to_reserve = max(0, self.min_reserved_pages - current_reserved)
                to_reserve = min(to_reserve, len(self.free_page_list), self.get_avail_physical_pages())
                if to_reserve <= 0:
                    continue

                pages_to_reserve = []

                for _ in range(to_reserve):
                    if self.free_page_list:
                        pages_to_reserve.append(self.free_page_list.popleft())
                    else:
                        break

            if pages_to_reserve:
                try:
                    self._map_pages(pages_to_reserve)
                    with self._lock:
                        self.reserved_page_list.extend(pages_to_reserve)
                        self._update_memory_usage()
                        self._cond.notify_all()
                    logger.debug(
                        "Preallocated %d pages, reserved=%d", len(pages_to_reserve), len(self.reserved_page_list)
                    )
                except Exception as e:
                    with self._lock:
                        self.free_page_list.extendleft(pages_to_reserve)
                        self._cond.notify_all()
                    logger.error("Failed to preallocate %d pages: %s", len(pages_to_reserve), e)

    def _start_prealloc_thread(self):
        if self.prealloc_thd is None:
            self.prealloc_running = True
            self.prealloc_thd = threading.Thread(target=self._prealloc_worker, daemon=True)
            self.prealloc_thd.start()

            self._trigger_preallocation()

    def _stop_prealloc_thread(self, timeout: float | None = None):
        if self.prealloc_thd is not None:
            with self._lock:
                self.prealloc_running = False
                self._cond.notify_all()
            self.prealloc_thd.join(timeout)
            if self.prealloc_thd.is_alive():
                logger.warning("Preallocation thread did not stop within timeout")
            self.prealloc_thd = None
            logger.debug("Stopped page preallocation thread")

    def _trigger_preallocation(self):
        """Trigger the preallocation thread to fill up reserved blocks"""
        with self._lock:
            self.prealloc_needed = True
            self._cond.notify_all()

    def _map_pages(self, page_ids: list[int]) -> None:
        if not _vmm_ops_available:
            return

        if self.contiguous_layout:
            offsets = [pid * self.page_size * self.num_layers * 2 for pid in page_ids]
        else:
            offsets = [pid * self.page_size for pid in page_ids]
        if self.tp_size > 1:
            broadcast_map_to_kv_tensors(self.tp_size, offsets)
        else:
            vmm_ops.map_to_kv_tensors(offsets)

    def _unmap_pages(self, page_ids: list[int]) -> None:
        if not _vmm_ops_available:
            return

        if self.contiguous_layout:
            offsets = [pid * self.page_size * self.num_layers * 2 for pid in page_ids]
        else:
            offsets = [pid * self.page_size for pid in page_ids]
        if self.tp_size > 1:
            broadcast_unmap_from_kv_tensors(self.tp_size, offsets)
        else:
            if self.async_sched:
                torch.cuda.synchronize()
            vmm_ops.unmap_from_kv_tensors(offsets)

    def _update_memory_usage(self):
        """Update memory usage information in shared memory."""
        used_phy_mem_size = self.get_num_inuse_pages() * self.num_layers * self.page_size * 2
        prealloc_phy_mem_size = self.get_num_reserved_pages() * self.num_layers * self.page_size * 2

        self.mem_info_tracker.update_memory_usage(used_size=used_phy_mem_size, prealloc_size=prealloc_phy_mem_size)
