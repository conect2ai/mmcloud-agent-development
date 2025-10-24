def sanitize_cell(val) -> str:
    """
    Ensure CSV-friendly text: stringified, no newlines, trimmed.
    Keeps everything on one line to avoid 'broken' rows in spreadsheet apps.
    """
    if val is None:
        return ""
    s = str(val)
    # collapse newlines/tabs
    s = s.replace("\r\n", " ").replace("\n", " ").replace("\r", " ").replace("\t", " ")
    # optional: collapse multiple spaces
    while "  " in s:
        s = s.replace("  ", " ")
    return s.strip()