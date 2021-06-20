import setuptools
from pensimpy._version import __version__

with open("README.md", "r") as fh:
    long_description = fh.read()


with open("requirements.txt", "r") as fh:
    requirements = fh.read()

setuptools.setup(
    name="pensimpy", 
    version=__version__,
    author="Quartic",
    author_email="",
    description="The Python implementation of IndPenSim",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Quarticai/PenSimPy.git",
    packages=setuptools.find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
