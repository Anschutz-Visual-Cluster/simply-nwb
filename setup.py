from setuptools import setup, find_packages

VERSION = "1.4.4"


def parse_requirements(requirement_file):
    with open(requirement_file) as fi:
        return fi.readlines()


with open('./README.rst') as f:
    long_description = f.read()



setup(
    name='simply_nwb',
    packages=find_packages(),
    version=VERSION,
    description='Common NWB use cases and simplified interface for common usage',
    author='Spencer Hanson',
    long_description=long_description,
    install_requires=parse_requirements('requirements.txt'),
    keywords=['neuroscience', 'nwb', 'tools', 'science'],
    classifiers=[
        'Programming Language :: Python :: 3'
    ]
)

