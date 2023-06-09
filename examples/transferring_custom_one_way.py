from simply_nwb.transferring import OneWayFileSync


def remove_extension(filename):
    return filename.split(".")[0]


def main():
    OneWayFileSync(
        # Will watch the folder 'from_src' for files and changes
        source_directory="../data/from_src",
        # Will copy all new files / changes to this directory
        destination_directory="../data/to_dst",
        watch_file_glob={
            # Copy all bmp files
            "*.bmp": {},
            # Will copy all <name>.txt files to 'TextFiles/myfile_name_me_<name>.py'
            # Will create directories under the destination folder
            "*.txt": {
                "filename": "TextFiles/myfile_{name}.py",
                "name_func": remove_extension
            }
        },
        delete_on_copy=True # Will delete the file/folder from the src directory upon successful copy
    ).start()
    # Program will run continuously until killed


if __name__ == "__main__":
    main()
