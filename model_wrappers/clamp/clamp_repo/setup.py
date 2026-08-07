import os

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="clamp",
    py_modules=["clamp"],
    version="1.0",
    author="Philipp Seidl",
    author_email="ph.seidl92@gmail.com",
    packages=find_packages(exclude=["tests*"]),
    install_requires="""mhnreact@git+https://github.com/ml-jku/mhn-react.git@a3e0cc6fe3b8715f2e9b91352a0b66893b4aa560
        clip@git+https://github.com/openai/CLIP.git@d05afc436d78f1c48dc0dbf8e5980a9d471f35f6
        numpy
        tqdm
        scikit-learn
        scipy
        pandas
        nltk
        rdkit-pypi
        transformers
        matplotlib
        torch
        torchvision
        loguru
        wandb
        mlflow
        swifter""".split(),
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ml-jku/clamp",
    python_requires='>=3.7',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD-2-Clause License",
        "Operating System :: linux-64",
    ],
    include_package_data=True,
    extras_require={'dev': ['pytest']},
)
