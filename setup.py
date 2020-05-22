from distutils.core import setup

from setuptools import find_packages
from pensimpy._version import __version__


def get_requirements(path):
    requirements = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            requirements.append(line)

    return requirements


setup(
    name='pensimpy',
    version=__version__,
    packages=find_packages(exclude=["*data*", "speed_benchmarks"]),
    license='',
    author='Quartic.ai Data Science Team',
    install_requires=get_requirements('requirements.txt'),
    long_description=open('README.md').read(),
)
