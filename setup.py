from setuptools import setup, find_packages
from typing import List


# =========================
# CONSTANTS
# =========================
PROJECT_NAME = "heart-disease-mlops"
VERSION = "0.0.1"
AUTHOR = "Haruna Sani"
DESCRIPTION = "End-to-end Heart Disease Classification System"


# =========================
# REQUIREMENTS LOADER
# =========================
def get_requirements(file_path: str) -> List[str]:
    """
    Reads requirements.txt and returns list of dependencies.
    
    Parameters
    ----------
    file_path : str
        Path to requirements.txt file

    Returns
    -------
    List[str]
        List of packages
    """
    requirements = []

    try:
        with open(file_path, "r") as file:
            requirements = file.readlines()
            requirements = [req.replace("\n", "") for req in requirements]

            # Remove empty lines and comments
            requirements = [
                req for req in requirements
                if req and not req.startswith("#")
            ]

    except FileNotFoundError:
        print("requirements.txt not found")

    return requirements


# =========================
# SETUP CONFIGURATION
# =========================
setup(
    name=PROJECT_NAME,
    version=VERSION,
    author=AUTHOR,
    description=DESCRIPTION,
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
    python_requires=">=3.8"
)