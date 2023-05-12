from setuptools import setup

VERSION = "0.0.3"


def parse_requirements(requirement_file):
    with open(requirement_file) as fi:
        return fi.readlines()


with open('./README.rst') as f:
    long_description = f.read()


setup(
    name='simply_nwb',
    packages=['simply_nwb', 'simply_nwb.acquisition', 'simply_nwb.acquisition.tools', 'simply_nwb.transforms'],
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

