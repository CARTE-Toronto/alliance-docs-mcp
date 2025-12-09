---
title: "Anaconda/en"
url: "https://docs.alliancecan.ca/wiki/Anaconda/en"
category: "General"
last_modified: "2025-12-02T14:59:54Z"
page_id: 4505
display_title: "Anaconda"
---

`<languages />`{=html}

Anaconda is a Python distribution.

`<span id="Pourquoi_Anaconda_n&#039;est_pas_recommandé_sur_une_grappe_de_calcul_?">`{=html}`</span>`{=html}

# Why is Anaconda not recommended on a cluster ? {#why_is_anaconda_not_recommended_on_a_cluster}

Anaconda may cause issues on a cluster for multiple reasons:

- Anaconda very often installs software (compilers, scientific libraries etc.) which already exist on our clusters as modules, with a configuration that is not optimal, and which may cause conflicts.
- It installs binaries which are not optimized for the processor architecture on our clusters. Your jobs may be slower because of it.
- It makes incorrect assumptions about the location of various system libraries. Your jobs may encounter errors when running.
- Anaconda uses the `$HOME` directory for its installation, where it writes an enormous number of files. A single Anaconda installation can easily absorb almost half of your quota for the number of files in your home directory.
- Anaconda is slower than the installation of packages via Python wheels.
- Anaconda modifies the `$HOME/.bashrc` file, which can easily cause conflicts.

# What are alternatives ? {#what_are_alternatives}

The first step you should take is to contact our [Technical support](https://docs.alliancecan.ca/Technical_support "Technical support"){.wikilink}, so that our experts investigate with your what is the best alternative for your needs. If you prefer to attempt it yourself, two main options are listed below.

`<span id="Transitionner_de_Conda_vers_virtualenv">`{=html}`</span>`{=html}

## Transition from Conda to virtualenv {#transition_from_conda_to_virtualenv}

A [virtual environment](https://docs.alliancecan.ca/Python#Creating_and_using_a_virtual_environment "virtual environment"){.wikilink} offers you all the functionality which you need to use Python on our clusters. This should be the first option that you explore. Here is how to convert to the use of virtual environments if you use Anaconda on your personal computer:

1.  List the dependencies (requirements) of the application you want to use. To do so, you can:
    1.  Run `pip show ``<package_name>`{=html} from your virtual environment (if the package exists on [PyPI](https://pypi.org/))
    2.  Or, check if there is a `requirements.txt` file in the Git repository.
    3.  Or, check the variable `install_requires` of the file `setup.py`, which lists the requirements.
2.  Find which dependencies are Python modules and which are libraries provided by Anaconda. For example, CUDA and CuDNN are libraries which are available on Anaconda Cloud but which you should not install yourself on our clusters - they are already installed.
3.  Remove from the list of dependencies everything which is not a Python module (e.g. `cudatoolkit` and `cudnn`).
4.  Use a [virtual environment](https://docs.alliancecan.ca/Python#Creating_and_using_a_virtual_environment "virtual environment"){.wikilink} in which you will install your dependencies.

Your software should run - if it doesn\'t, don\'t hesitate to [contact us](https://docs.alliancecan.ca/Technical_support "contact us"){.wikilink}.

`<span id="Utiliser_Apptainer">`{=html}`</span>`{=html}

## Using Apptainer {#using_apptainer}

In some situations, the complexity of the dependencies of a program requires the use of a solution where you can control the entire software environment. In these situations, we recommend the tool [ Apptainer](https://docs.alliancecan.ca/Apptainer#Using_Conda_in_Apptainer " Apptainer"){.wikilink}; note that a Docker image can be converted into an Apptainer image. The only disadvantage of Apptainer is its consumption of disk space. If your research group plans on using several images, it would be wise to collect all of them together in a single directory of the group\'s project space to avoid duplication.
