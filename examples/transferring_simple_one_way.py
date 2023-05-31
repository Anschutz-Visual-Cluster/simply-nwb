from simply_nwb.transferring import OneWayFileSync


def main():
    OneWayFileSync(
        # Will watch the folder 'from_src' for files and changes
        "../data/from_src",
        # Will copy all new files / changes to this directory
        "../data/to_dst",
        # Include all files, could also do '*.txt' for all txt files, etc
        "*",
        delete_on_copy=True  # Will delete the file/folder from the src directory upon successful copy
    ).start()
    # Program will run continuously until killed


if __name__ == "__main__":
    main()

