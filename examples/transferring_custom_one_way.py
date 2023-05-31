from simply_nwb.transferring import OneWayFileSync


def remove_extension(filename):
    return filename.split(".")[0]


def main():
    OneWayFileSync(
        # Will watch the folder 'from_src' for files and changes
        "../data/from_src",
        # Will copy all new files / changes to this directory
        "../data/to_dst",
        # asdf
        {
            "*.bmp": {},
            # Will copy all <name>.txt files to 'TextFiles/myfile_name_me_<name>.py'
            # Will create directories under the destination folder
            "*.txt": {
                "filename": "TextFiles/myfile_{name}.py",
                "name_func": remove_extension
            }
        },
        delete_on_copy=True  # Will delete the file/folder from the src directory upon successful copy
    ).start()
    # Program will run continuously until killed


if __name__ == "__main__":
    main()

