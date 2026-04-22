# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Aphrodite project
def sanitize_filename(filename: str) -> str:
    return filename.replace("/", "_").replace("..", "__").strip("'").strip('"')
