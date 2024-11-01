from setuptools import sandbox
# On linux you might need to do sudo apt-get install python3-setuptools


def main():
    # Call setup.py from python
    sandbox.run_setup('setup.py', [
        'clean',  # Clean the build directories
        'bdist_wheel',  # Include a prebuilt wheel
        'sdist'  # Include a source dist
    ])


if __name__ == "__main__":
    main()
