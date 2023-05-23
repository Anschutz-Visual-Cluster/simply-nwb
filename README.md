# simply-nwb

### ReadTheDocs [Link](https://simply-nwb.readthedocs.io/en/latest/index.html)

### Publishing a new version
- Update the version number in `setup.py` try to use [sem ver](https://semver.org/) as a guide for which number to bump
- Run `build_package_script.py` to build a new version of the package
- Make sure your dist/ folder contains only the new version (could fail if not!)
- Run `publish_package.py` to upload the contents to the dist/ folder to pypi

### Notable Repository Files
#### `publish_package.py`
- Run this script to publish the latest built package
- Will attempt to publish all packages in `dist/` so make sure you run `build_package_script.py` first

#### `build_docs_script.py`
- Run this script to generate the docs
- TODO more details

#### `build_package_script.py`
- Run this script to build a new version of the package
- Make sure you increment the version in `setup.py` to generate the correct version
- Use [sem ver](https://semver.org/) as a guide when changing the version

#### `setup.py`
- Contains all the package metadata, including version

#### `requirements.txt`
- requirements for the package
- install with `pip install -r requirements.txt`

#### `dev-requirements.txt`
- requirements to develop the package
- includes test libraries, doc generation libs
- install with `pip install -r dev-requirements.txt`

#### `README.rst`
- Document for the package description on PyPi