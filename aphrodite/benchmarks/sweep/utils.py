def sanitize_filename(filename: str) -> str:
    return filename.replace("/", "_").replace("..", "__").strip("'").strip('"')
