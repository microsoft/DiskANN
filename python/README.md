# diskannpy

## Installation
Packages published to PyPI will always be built using the latest numpy major.minor release (at this time, 1.25).

Conda distributions for versions 1.19-1.25 will be completed as a future effort.  In the meantime, feel free to 
clone this repository and build it yourself.

## Local Build Instructions
Please see the [Project README](https://github.com/microsoft/DiskANN/blob/main/README.md) for system dependencies and requirements.

After ensuring you've followed the directions to build the project library and executables, you will be ready to also
build `diskannpy` with these additional instructions.

### Changing Numpy Version
In the root folder of DiskANN, there is a file `pyproject.toml`. You will need to edit the version of numpy in both the
`[build-system.requires]` section, as well as the `[project.dependencies]` section.  The version numbers must match.

```bash
python3.11 -m venv venv # versions from python3.8 and up should work. on windows, you might need to use py -3.11 -m venv venv
source venv/bin/activate # linux
# or
venv\Scripts\Activate.{ps1, bat} # windows
pip install build
python -m build
```

The built wheel will be placed in the `dist` directory in your DiskANN root. Install it using `pip install dist/<wheel name>.whl`

## Citations
Please cite this software in your work as:
```
@misc{diskann-github,
   author = {Simhadri, Harsha Vardhan and Krishnaswamy, Ravishankar and Srinivasa, Gopal and Subramanya, Suhas Jayaram and Antonijevic, Andrija and Pryce, Dax and Kaczynski, David and Williams, Shane and Gollapudi, Siddarth and Sivashankar, Varun and Karia, Neel and Singh, Aditi and Jaiswal, Shikhar and Mahapatro, Neelam and Adams, Philip and Tower, Bryan}},
   title = {{DiskANN: Graph-structured Indices for Scalable, Fast, Fresh and Filtered Approximate Nearest Neighbor Search}},
   url = {https://github.com/Microsoft/DiskANN},
   version = {0.5},
   year = {2023}
}
```