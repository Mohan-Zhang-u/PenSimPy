import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pensimpy", 
    version="0.0.1",
    author="Quartic",
    author_email="",
    description="Pckage to simulate penicillin yield",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Quarticai/PenSimPy.git",
    packages=setuptools.find_packages(),
    install_requires=['numpy==1.18.2', 'matplotlib==3.2.1', 'pandas==1.0.3', 'scipy==1.4.1', 'tqdm==4.45.0', 'fastodeint==0.0.1'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
