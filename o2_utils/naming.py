

def to_snake_case(string):
    """Convert a string to snake case (lowercase, and underscores instead of spaces or dashes)
    """
    string = string.strip('\n\r\t ')
    string = string.replace('-', '_')
    string = string.replace(' ', '_')
    string = string.lower()
    return string


def filename_from_path(path):
    """Returns just the file name (no extension) from a full path (eg /my/path/to/file.txt returns 'file')

    Arguments:
        path {Path} -- full path

    Returns:
        str -- the file name without extension
    """
    return os.path.splitext(os.path.basename(path))[0]