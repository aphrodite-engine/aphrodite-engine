from aphrodite.triton_utils.importing import (HAS_TRITON,
                                              TritonLanguagePlaceholder,
                                              TritonLibdevicePlaceholder,
                                              TritonPlaceholder)

if HAS_TRITON:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
else:
    triton = TritonPlaceholder()
    tl = TritonLanguagePlaceholder()
    libdevice = TritonLibdevicePlaceholder()

__all__ = ["HAS_TRITON", "triton", "tl"]
