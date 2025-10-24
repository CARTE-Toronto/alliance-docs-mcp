---
title: "Anaconda/en"
url: "https://docs.alliancecan.ca/wiki/Anaconda/en"
category: "General"
last_modified: "2025-02-13T20:44:32Z"
page_id: 4505
display_title: "Anaconda"
---

`<languages />`{=html}

Anaconda is a Python distribution. We ask our users to **not install Anaconda on our clusters**. We recommend that you consider other options like a virtual environment or a [Apptainer](https://docs.alliancecan.ca/Apptainer "Apptainer"){.wikilink} container, for the most complicated cases.

## Do not install Anaconda on our clusters {#do_not_install_anaconda_on_our_clusters}

We are aware of the fact that Anaconda is widely used in several domains, such as data science, AI, bioinformatics etc. Anaconda is a useful solution for simplifying the management of Python and scientific libraries on a personal computer. However, on a cluster like those supported by the Alliance, the management of these libraries and dependencies should be done by our staff, in order to ensure compatibility and optimal performance. Here is a list of reasons:

- Anaconda very often installs software (compilers, scientific libraries etc.) which already exist on our clusters as modules, with a configuration that is not optimal.
- It installs binaries which are not optimized for the processor architecture on our clusters.
- It makes incorrect assumptions about the location of various system libraries.
- Anaconda uses the `$HOME` directory for its installation, where it writes an enormous number of files. A single Anaconda installation can easily absorb almost half of your quota for the number of files in your home directory.
- Anaconda is slower than the installation of packages via Python wheels.
- Anaconda modifies the `$HOME/.bashrc` file, which can easily cause conflicts.

## How to transition from Conda to virtualenv {#how_to_transition_from_conda_to_virtualenv}

A [virtual environment](https://docs.alliancecan.ca/Python#Creating_and_using_a_virtual_environment "virtual environment"){.wikilink} offers you all the functionality which you need to use Python on our clusters. Here is how to convert to the use of virtual environments if you use Anaconda on your personal computer:

1.  List the dependencies (requirements) of the application you want to use. To do so, you can:
    1.  Run `pip show ``<package_name>`{=html} from your virtual environment (if the package exists on [PyPI](https://pypi.org/))
    2.  Or, check if there is a `requirements.txt` file in the Git repository.
    3.  Or, check the variable `install_requires` of the file `setup.py`, which lists the requirements.
2.  Find which dependencies are Python modules and which are libraries provided by Anaconda. For example, CUDA and CuDNN are libraries which are available on Anaconda Cloud but which you should not install yourself on our clusters - they are already installed.
3.  Remove from the list of dependencies everything which is not a Python module (e.g. `cudatoolkit` and `cudnn`).
4.  Use a [virtual environment](https://docs.alliancecan.ca/Python#Creating_and_using_a_virtual_environment "virtual environment"){.wikilink} in which you will install your dependencies.

Your software should run - if it doesn\'t, don\'t hesitate to [contact us](https://docs.alliancecan.ca/Technical_support "contact us"){.wikilink}.

## Apptainer Use {#apptainer_use}

In some situations, the complexity of the dependencies of a program requires the use of a solution where you can control the entire software environment. In these situations, we recommend the tool [ Apptainer](https://docs.alliancecan.ca/Apptainer#Using_Conda_in_Apptainer " Apptainer"){.wikilink}; note that a Docker image can be converted into an Apptainer image. The only disadvantage of Apptainer is its consumption of disk space. If your research group plans on using several images, it would be wise to collect all of them together in a single directory of the group\'s project space to avoid duplication.

## Examples where Anaconda does not work {#examples_where_anaconda_does_not_work}

R
:   A conda recipe forces the installation of R. This installation does not perform nearly as well as the version we provide as a module (which uses Intel MKL). This same R does not work well, and jobs launched with it may die and waste both computing resources as well as your time.
