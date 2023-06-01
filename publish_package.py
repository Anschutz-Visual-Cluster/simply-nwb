import sys
import os
from twine.__main__ import main as twine_main


def main():
    pypirc_file = os.path.expanduser('~/.pypirc')
    if not os.path.exists(pypirc_file):
        # See https://packaging.python.org/en/latest/specifications/pypirc/ for help
        raise ValueError("Must have a .pypirc file to upload a package to PyPi! See https://packaging.python.org/en/latest/specifications/pypirc/ for more information")

    sys.argv = [
        sys.argv[0],
        "upload",
        "-r",
        "pypi",
        "dist/*"
    ]
    twine_main()
    pass


if __name__ == "__main__":
    main()
