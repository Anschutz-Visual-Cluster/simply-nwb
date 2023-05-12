from simply_nwb.util import inspect_nwb_file
from nwbinspector.inspector_tools import format_messages
import sys


def main():
    filename_arg = " ".join(sys.argv[1:])
    print(f"Inspecting file: '{filename_arg}'")
    res = inspect_nwb_file(filename_arg)
    print("\n".join(format_messages(res, levels=["importance", "file_path"])))
    if not res:
        print("PASSED!")


if __name__ == "__main__":
    main()

