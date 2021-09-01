"""Base module variables."""
import importlib.util
import json
import os.path as op
from pathlib import Path

# Get version
spec = importlib.util.spec_from_file_location(
    "_version", op.join(op.dirname(__file__), "ddmra/_version.py")
)
_version = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_version)

VERSION = _version.get_versions()["version"]
del _version

# Get list of authors from Zenodo file
with open(op.join(op.dirname(__file__), ".zenodo.json"), "r") as fo:
    zenodo_info = json.load(fo)
authors = [author["name"] for author in zenodo_info["creators"]]
author_names = []
for author in authors:
    if ", " in author:
        author_names.append(author.split(", ")[1] + " " + author.split(", ")[0])
    else:
        author_names.append(author)

# Get package description from README
# Since this file is executed from ../setup.py, the path to the README is determined by the
# location of setup.py.
readme_path = Path(__file__).parent.joinpath("README.md")
longdesc = readme_path.open().read()

# Fields
AUTHOR = "Taylor Salo"
COPYRIGHT = "Copyright 2021, Taylor Salo"
CREDITS = author_names
LICENSE = "LGPL 2.1"
MAINTAINER = "Taylor Salo"
EMAIL = "tsalo006@fiu.edu"
STATUS = "Prototype"
URL = "https://github.com/ME-ICA/ddmra"
PACKAGENAME = "ddmra"
DESCRIPTION = "A Python package for distance-dependent motion-related artifact (DDMRA) analysis."
LONGDESC = longdesc

DOWNLOAD_URL = "https://github.com/tsalo/{name}/archive/{ver}.tar.gz".format(
    name=PACKAGENAME, ver=VERSION
)

REQUIRES = [
    "joblib",
    "nibabel",
    "nilearn",
    "numpy>=1.15",
    "scipy>=1.3.3",
    "seaborn",
]

TESTS_REQUIRES = [
    "codecov",
    "coverage<5.0",
    "flake8>=3.7",
    "flake8-black",
    "flake8-isort",
    "pytest",
    "pytest-cov",
]

EXTRA_REQUIRES = {
    "dev": ["versioneer"],
    "doc": [
        "nbsphinx",
        "sphinx>=1.5.3",
        "sphinx_rtd_theme",
    ],
    "tests": TESTS_REQUIRES,
    "duecredit": ["duecredit"],
}

ENTRY_POINTS = {}

# Enable a handle to install all extra dependencies at once
EXTRA_REQUIRES["all"] = list(set([v for deps in EXTRA_REQUIRES.values() for v in deps]))

# Supported Python versions using PEP 440 version specifiers
# Should match the same set of Python versions as classifiers
PYTHON_REQUIRES = ">=3.6"

# Package classifiers
CLASSIFIERS = [
    "Development Status :: 2 - Pre-Alpha",
    "Intended Audience :: Science/Research",
    "Topic :: Scientific/Engineering :: Information Analysis",
    "License :: OSI Approved :: GNU Library or Lesser General Public License (LGPL)",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
]
