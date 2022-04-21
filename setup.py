import setuptools

with open("README.md", "r") as fh:
  long_description = fh.read()

setuptools.setup(
    name="usc_learning",
    version="0.0.1",
    author="guillaume10",
    author_email="bellegar@usc.edu",
    description="A framework for running reinforcement learning and imitation learning (mainly on quadrupeds).",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)