from aphrodite.triton_utils.importing import HAS_TRITON, TritonLanguagePlaceholder, TritonPlaceholder

if HAS_TRITON:
    import triton
    import triton.language as tl
    import triton.language.extra.libdevice as tldevice
else:
    triton = TritonPlaceholder()
    tl = TritonLanguagePlaceholder()
    tldevice = TritonLanguagePlaceholder()

__all__ = ["HAS_TRITON", "triton", "tl", "tldevice"]
