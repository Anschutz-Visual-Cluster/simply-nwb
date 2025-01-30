import sys
import os
from twine.__main__ import main as twine_main


def main():
    """
    If your upload is giving HTTPError: 403 Forbidden ...
    You need to format your .pypirc file like the following..

    [distutils]
    index-servers =
      pypi
      graph_judge

    [pypi]
    username = __token__
    password = pypi-...<default pypi token you want to use>


    [simply-nwb]
    repository = https://upload.pypi.org/legacy/
    username = __token__
    password = pypi-...<project specific token here>
    """


    pypirc_file = os.path.expanduser('~/.pypirc')
    if not os.path.exists(pypirc_file):
        # See https://packaging.python.org/en/latest/specifications/pypirc/ for help
        raise ValueError("Must have a .pypirc file to upload a package to PyPi! See https://packaging.python.org/en/latest/specifications/pypirc/ for more information")

    sys.argv = [
        sys.argv[0],
        "upload",
        "-r",
        "simply_nwb",  # dont forget to change me!
        "dist/*",
    ]
    twine_main()


if __name__ == "__main__":
    main()
