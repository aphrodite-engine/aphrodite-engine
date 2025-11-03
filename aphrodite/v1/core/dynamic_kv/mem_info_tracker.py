import atexit
import os
import signal
from typing import Optional

import posix_ipc

from .cli_utils import (MemInfoStruct, RwLockedShm, get_ipc_name, get_ipc_path,
                        init_kv_cache_limit)
from .utils import DEFAULT_IPC_NAME


class MemInfoTracker:
    """Tracks memory usage information through shared memory."""

    def __init__(self, total_mem_size: int):
        """
        Args:
            total_mem_size: Total memory size to initialize shared memory with
        """
        self.ipc_name = get_ipc_name(DEFAULT_IPC_NAME)
        init_kv_cache_limit(self.ipc_name, total_mem_size)
        self._register_cleanup()

    def check_and_get_resize_target(self, current_mem_size: int,
                                    num_layers: int) -> Optional[int]:
        """Check if memory size has changed and return new target size if needed."""
        with RwLockedShm(self.ipc_name, MemInfoStruct.SHM_SIZE,
                         RwLockedShm.RLOCK) as mm:
            mem_info = MemInfoStruct.from_buffer(mm)
            new_mem_size = mem_info.total_size // num_layers // 2
            if new_mem_size != current_mem_size:
                return new_mem_size
        return None

    def update_memory_usage(self, used_size: int, prealloc_size: int):
        """Update the memory usage information in shared memory."""
        with RwLockedShm(self.ipc_name, MemInfoStruct.SHM_SIZE,
                         RwLockedShm.WLOCK) as mm:
            mem_info = MemInfoStruct.from_buffer(mm)
            mem_info.used_size = used_size
            mem_info.prealloc_size = prealloc_size
            mem_info.write_to_buffer(mm)

    def cleanup(self, *args):
        """Remove the POSIX shared-memory segment and its backing file."""
        try:
            posix_ipc.unlink_shared_memory(self.ipc_name)
        except Exception:
            pass

        try:
            os.unlink(get_ipc_path(self.ipc_name))
        except FileNotFoundError:
            pass

        if args and isinstance(args[0], int):
            signum = args[0]
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)

    def _register_cleanup(self):
        """Register atexit and signal handlers for shared-memory cleanup."""
        atexit.register(self.cleanup)

        for _sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP,
                     signal.SIGQUIT):
            try:
                signal.signal(_sig, self.cleanup)
            except Exception:
                pass

