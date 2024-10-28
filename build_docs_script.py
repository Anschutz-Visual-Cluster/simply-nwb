import os
import platform
import subprocess


def run_make(cmd):
    if platform.system().startswith("Windows"):
        subprocess.run(["make.bat", cmd])
    else:
        try:
            subprocess.run(["make", cmd])
        except Exception as e:
            print("Error running make, is it installed? apt-get install make")
            raise e


def main():
    os.chdir("docs")

    run_make("clean")

    print("Generating API docs..")
    subprocess.run(["sphinx-apidoc", "-o", "source",  "../simply_nwb", "-f"])

    print("Generating HTML docs..")

    subprocess.run(["sphinx-build",  "-b", "html", "source", "build"])

    print("Making HTML docs..")

    run_make("html")

    print("Done!")


if __name__ == "__main__":
    main()
