# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from enum import Enum


class SPDXStatus(Enum):
    EMPTY = "empty"
    COMPLETE = "complete"
    MISSING_LICENSE = "missing_license"
    MISSING_COPYRIGHT = "missing_copyright"
    MISSING_BOTH = "missing_both"


LICENSE_LINE = "# SPDX-License-Identifier: Apache-2.0"
COPYRIGHT_LINES = {
    "# SPDX-FileCopyrightText: Copyright contributors to the vLLM project",
}
DEFAULT_COPYRIGHT_LINE = "# SPDX-FileCopyrightText: Copyright contributors to the vLLM project"


def check_spdx_header_status(file_path: str) -> SPDXStatus:
    with open(file_path, encoding="utf-8") as file:
        lines = file.readlines()
        if not lines:
            return SPDXStatus.EMPTY

        start_idx = 1 if lines and lines[0].startswith("#!") else 0
        has_license = False
        has_copyright = False

        for line in lines[start_idx:]:
            stripped = line.strip()
            if stripped == LICENSE_LINE:
                has_license = True
            elif stripped in COPYRIGHT_LINES:
                has_copyright = True

        if has_license and has_copyright:
            return SPDXStatus.COMPLETE
        if has_license:
            return SPDXStatus.MISSING_COPYRIGHT
        if has_copyright:
            return SPDXStatus.MISSING_LICENSE
        return SPDXStatus.MISSING_BOTH


def add_header(file_path: str, status: SPDXStatus) -> None:
    with open(file_path, "r+", encoding="utf-8") as file:
        lines = file.readlines()
        file.seek(0, 0)
        file.truncate()

        if status == SPDXStatus.MISSING_BOTH:
            header = f"{LICENSE_LINE}\n{DEFAULT_COPYRIGHT_LINE}\n"
            if lines and lines[0].startswith("#!"):
                file.write(lines[0])
                file.write(header)
                file.writelines(lines[1:])
            else:
                file.write(header)
                file.writelines(lines)
            return

        if status == SPDXStatus.MISSING_COPYRIGHT:
            for i, line in enumerate(lines):
                if line.strip() == LICENSE_LINE:
                    lines.insert(i + 1, f"{DEFAULT_COPYRIGHT_LINE}\n")
                    break
            file.writelines(lines)
            return

        if status == SPDXStatus.MISSING_LICENSE:
            for i, line in enumerate(lines):
                if line.strip() in COPYRIGHT_LINES:
                    lines.insert(i, f"{LICENSE_LINE}\n")
                    break
            file.writelines(lines)


def main() -> int:
    files_to_fix: list[tuple[str, SPDXStatus]] = []
    for file_path in sys.argv[1:]:
        status = check_spdx_header_status(file_path)
        if status in {
            SPDXStatus.MISSING_BOTH,
            SPDXStatus.MISSING_COPYRIGHT,
            SPDXStatus.MISSING_LICENSE,
        }:
            files_to_fix.append((file_path, status))

    if files_to_fix:
        print("The following files are missing the SPDX header:")
        for file_path, status in files_to_fix:
            print(f"  {file_path}")
            add_header(file_path, status)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
