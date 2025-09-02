"""
Shared constants for Aphrodite endpoints.
"""

# HTTP header limits for h11 parser
# These constants help mitigate header abuse attacks
H11_MAX_INCOMPLETE_EVENT_SIZE_DEFAULT = 4194304  # 4 MB
H11_MAX_HEADER_COUNT_DEFAULT = 256
