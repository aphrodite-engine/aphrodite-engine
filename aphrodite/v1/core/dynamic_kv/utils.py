import os


def _sanitize_segment(segment: str) -> str:
    """Sanitize a segment to safe characters for SHM names."""
    allowed = []
    for ch in segment:
        if ch.isalnum() or ch in ("_", "-"):
            allowed.append(ch)
        else:
            allowed.append("-")
    return "".join(allowed)[:64]

def _ipc_segment_exists(name: str) -> bool:
    """Return True if a shared-memory segment/file with this name exists."""
    try:
        return os.path.exists(os.path.join(SHM_DIR, name))
    except Exception:
        return False


def _obtain_default_ipc_name() -> str:
    """Return a default IPC name like kvcached_<Engine>_<PGID>."""
    engine_tag = "aphrodite"
    try:
        group_id = os.getpgid(0)
    except Exception:
        try:
            group_id = os.getsid(0)
        except Exception:
            group_id = os.getpid()

    explicit = os.getenv("KVCACHED_IPC_NAME")
    if explicit:
        preferred = _sanitize_segment(explicit)

        if not _ipc_segment_exists(preferred):
            return preferred

        base_candidate = f"{preferred}_{engine_tag}_{group_id}"
        if not _ipc_segment_exists(base_candidate):
            return base_candidate
        for i in range(1, 100):
            candidate = f"{base_candidate}_{i}"
            if not _ipc_segment_exists(candidate):
                return candidate
        return f"{base_candidate}_{os.getpid()}"

    base = "kvcached"
    name = f"{base}_{engine_tag}_{group_id}"
    if not _ipc_segment_exists(name):
        return name
    for i in range(1, 100):
        candidate = f"{name}_{i}"
        if not _ipc_segment_exists(candidate):
            return candidate
    return f"{name}_{os.getpid()}"


def _get_page_size() -> int:
    """Get PAGE_SIZE from environment variable with validation."""
    default_page_size = 2 * 1024 * 1024
    page_size_mb_str = os.getenv("KVCACHED_PAGE_SIZE_MB")

    if page_size_mb_str is None:
        return default_page_size

    try:
        page_size = int(page_size_mb_str) * 1024 * 1024
    except ValueError:
        raise ValueError(
            f"Invalid KVCACHED_PAGE_SIZE_MB: {page_size_mb_str}. Must be an integer."
        )

    base_size = 2 * 1024 * 1024
    if page_size <= 0 or page_size % base_size != 0:
        raise ValueError(
            f"PAGE_SIZE must be a positive multiple of 2MB (2097152 bytes), "
            f"got: {page_size}")

    return page_size


PAGE_SIZE = _get_page_size()

GPU_UTILIZATION = float(os.getenv("KVCACHED_GPU_UTILIZATION", "0.95"))
PAGE_PREALLOC_ENABLED = os.getenv("KVCACHED_PAGE_PREALLOC_ENABLED",
                                  "true").lower() == "true"
MIN_RESERVED_PAGES = int(os.getenv("KVCACHED_MIN_RESERVED_PAGES", "5"))
MAX_RESERVED_PAGES = int(os.getenv("KVCACHED_MAX_RESERVED_PAGES", "10"))
SANITY_CHECK = os.getenv("KVCACHED_SANITY_CHECK", "false").lower() == "true"
CONTIGUOUS_LAYOUT = os.getenv("KVCACHED_CONTIGUOUS_LAYOUT",
                              "true").lower() == "true"

DEFAULT_IPC_NAME = _obtain_default_ipc_name()
SHM_DIR = "/dev/shm"


def align_to(x: int, a: int) -> int:
    return (x + a - 1) // a * a


def align_up_to_page(n_cells: int, cell_size: int) -> int:
    n_cells_per_page = PAGE_SIZE // cell_size
    aligned_n_cells = align_to(n_cells, n_cells_per_page)
    return aligned_n_cells

