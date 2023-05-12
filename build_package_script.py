from setuptools import sandbox


def main():
    # Call setup.py from python
    sandbox.run_setup('setup.py', [
        'clean',  # Clean the build directories
        'bdist_wheel',  # Include a prebuilt wheel
        'sdist'  # Include a source dist
    ])


if __name__ == "__main__":
    main()
