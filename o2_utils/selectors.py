import os
import warnings
from glob import glob


class GreaterThanExpectedMatchingFileError(Exception):
    def __init__(self,*args,**kwargs):
        Exception.__init__(self,*args,**kwargs)

class NoMatchingFilesError(Exception):
    def __init__(self,*args,**kwargs):
        Exception.__init__(self,*args,**kwargs)


def find_files_from_pattern(path, pattern, exclude_patterns=None, n_expected=1, error_behav='raise', verbose=False):
    """Find a given number of files on a given path, matching a particular glob pattern. 
    Raises an error if wrong number of files match, unless error_behav='pass'.

    Parameters
    ----------
    path : str
        Path to check

    pattern : str
        Passed to glob.glob

    exclude_patterns : list(str), optional
        If any of these patterns is in found file names, exclude them

    n_expected : int, optional
        Number of files expected to match pattern (default: 1)

    error_behav : str, optional
        If 'raise', raises errors; if 'pass', return None for no files, or the whole list for multiple (default: "raise").

    verbose :  bool, optional
        If True, print more verbose warnings.

    Raises
    ------
        GreaterThanExpectedMatchingFileError: If more than n_expected files match pattern
        NoMatchingFilesError: If no files match pattern

    Returns
    -------
    list(str) or None

    """
    if exclude_patterns is None:
        exclude_patterns = []
    elif isinstance(exclude_patterns, str):
        exclude_patterns = [exclude_patterns]

    if "*" not in pattern:
        warnings.warn(
            "It appears your glob pattern has no wildcard, are you sure that's right?"
        )

    potential_file_list = glob(f"{path}/{pattern}")

    # Raise error if no files found
    if len(potential_file_list) == 0:
        if error_behav == 'raise':
            raise NoMatchingFilesError(f"Found zero files matching {path}/{pattern}!")
        else:
            return None

    # Exclude requested patterns
    for exclude_pattern in exclude_patterns:
        potential_file_list = [
            pattern for pattern in potential_file_list if exclude_pattern not in os.path.basename(pattern)
        ]

    # If now no files, they were removed via pattern, so stop
    if len(potential_file_list) == 0:
        warnings.warn(f"No files remaining in {path} after exclusion.")
        if error_behav == 'raise':
            raise NoMatchingFilesError(f"Found zero files matching {path}/{pattern}!")
        else:
            return None

    # Raise error if still more than one matching file
    if len(potential_file_list) > n_expected:
        if error_behav == 'raise':
            if verbose:
                print("Found files:", potential_file_list)
            raise GreaterThanExpectedMatchingFileError(
                f"Found {len(potential_file_list)} files matching {path}/{pattern} but expected {n_expected}!"
            )
        else:
            return potential_file_list

    if n_expected == 1:
        return potential_file_list[0]
    else:
        return potential_file_list
