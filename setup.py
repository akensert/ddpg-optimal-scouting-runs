import setuptools
import os
import sys

def get_version():
  version_path = os.path.join(os.path.dirname(__file__), 'src')
  sys.path.insert(0, version_path)
  from _version import __version__ as version
  return version

with open("README.md", "r") as fh:
    long_description = fh.read()

install_requires = [
    "jupyter==1.0.0",
    "tensorflow==2.13.0",
    "scipy==1.11.3",
    "matplotlib==3.7.2",
    "tqdm==4.45.0",
    "gymnasium==0.26.2",
    "gymnasium[box2d]==0.26.2",
]

setuptools.setup(
    name='ddpg-optimal-scouting-runs',
    version=get_version(),
    author="Alexander Kensert",
    author_email="alexander.kensert@gmail.com",
    description="Implementation of a DDPG algorithm to perform scouting runs in chromatography.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    url="https://github.com/akensert/ddpg-optimal-scouting-runs",
    packages=setuptools.find_packages(include=["src*"]),
    install_requires=install_requires,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
    ],
    python_requires=">=3.10",
    keywords=[
        'tensorflow',
        'keras',
        'deep-learning',
        'machine-learning',
        'reinforcement-learning',
        'agent',
        'ddpg',
        'td3',
        'chromatography',
    ]
)
