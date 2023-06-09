from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="colortree",
    version="0.0.1",
    author="lisaalaz",
    author_email="",
    description="A package to build, visualize and evaluate colorful decision trees.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lisaalaz/colortree",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)