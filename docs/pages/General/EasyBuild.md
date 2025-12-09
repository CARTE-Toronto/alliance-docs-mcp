---
title: "EasyBuild/en"
url: "https://docs.alliancecan.ca/wiki/EasyBuild/en"
category: "General"
last_modified: "2023-09-14T16:12:01Z"
page_id: 18573
display_title: "EasyBuild"
---

`<languages />`{=html}

[EasyBuild](https://easybuild.io/) is a tool for building, installing, and maintaining software on high-performance computing systems. We use it to build almost everything in our software repository, [CVMFS](https://docs.alliancecan.ca/Accessing_CVMFS "CVMFS"){.wikilink}.

# EasyBuild and modules {#easybuild_and_modules}

One of the key features of EasyBuild is that it automatically generates environment [modules](https://docs.alliancecan.ca/Utiliser_des_modules/en "modules"){.wikilink} which can be used to make a software package available in your session. In addition to defining standard Linux environment variables such as `PATH`, `CPATH` and `LIBRARY_PATH`, EasyBuild also defines some environment variables specific to EasyBuild, two of which may be particularly interesting to users:

- `EBROOT``<name>`{=html}: Contains the full path to the location where the software `<name>`{=html} is installed.
- `EBVERSION``<name>`{=html}: Contains the full version of the software `<name>`{=html} loaded by this module.

For example, the module `python/3.10.2` on Narval defines:

- `EBROOTPYTHON`: `/cvmfs/soft.computecanada.ca/easybuild/software/2020/avx2/Core/python/3.10.2`
- `EBVERSIONPYTHON`: `3.10.2`

You can see the environment variables defined by the `python/3.10.2` module using:

`grep EB}}`

# Installation recipes and logs {#installation_recipes_and_logs}

EasyBuild keeps a copy of the recipe used to install each software package, as well as a detailed log inside the installation directory. This is accessible in the directory `$EBROOT``<name>`{=html}`/easybuild`. For example, for the `python/3.10.2` module, the installation directory contains, amongst other things:

- `$EBROOTPYTHON/easybuild/Python-3.10.2.eb`
- `$EBROOTPYTHON/easybuild/easybuild-Python-3.10.2-*.log`

# Using EasyBuild in your own account {#using_easybuild_in_your_own_account}

EasyBuild can be used to install software packages in your own account. However, in most cases, it is preferable to ask our [technical support](https://docs.alliancecan.ca/technical_support "technical support"){.wikilink} to install the software centrally for you. This is because that will ensure that the software package is available on all of our clusters. It will also avoid using your quota, and it will avoid causing undue load on the parallel filesystems.

## What is a recipe {#what_is_a_recipe}

Recipes, also known as EasyConfig files are text files containing the information EasyBuild needs to build a particular piece of software in a particular environment. They are named following a convention:

- `<name>`{=html}`-``<version>`{=html}`-``<toolchain name>`{=html}`-``<toolchain version>`{=html}`.eb`

where `<name>`{=html} is the name of the package, `<version>`{=html} is its version, `<toolchain name>`{=html} is the name of the toolchain and `<toolchain version>`{=html} is its version. More on toolchains later.

## Finding a recipe {#finding_a_recipe}

EasyBuild contains a lot of recipes which may or may not compile with the toolchains we have. The surest way to get a recipe that works is to start from one of the recipes which we have installed. These can be found either in the installation folder, as mentioned above, or in the `/cvmfs/soft.computecanada.ca/easybuild/ebfiles_repo/$EBVERSIONGENTOO` folder.

## Installing a software with EasyBuild {#installing_a_software_with_easybuild}

Once you have found a recipe matching your needs, copy its recipe from the `/cvmfs/soft.computecanada.ca/easybuild/ebfiles_repo/$EBVERSIONGENTOO` folder, and modify it as needed. Then, run

to install it. This will install the software inside of your home directory, in `$HOME/.local/easybuild`. After the installation is completed, exit your session and reconnect to the cluster, and it should be available to load as a module.

### Reinstalling an existing version {#reinstalling_an_existing_version}

If you are reinstalling the exact same version as one we have installed centrally, but with modified parameters, you need to use

to install a local version in your home.

### Installing in a different location {#installing_in_a_different_location}

You may want to install the software package in a different location than your home directory, for example in a project directory. To do so, use the following:

Then, to get these modules available in your sessions, run /path/to/your/project/easybuild/modules}} If you want to have this available by default in your sessions, you can add this command to your `.bashrc` file in your home.

# Additional resources {#additional_resources}

- Webinar [`<i>`{=html}Building software on Compute Canada clusters using EasyBuild`</i>`{=html}](https://westgrid.github.io/trainingMaterials/getting-started/#building-software-with-easybuild) (recording and slides)
- Our staff-facing documentation [is available here](https://github.com/ComputeCanada/software-stack/blob/main/doc/easybuild.md).
- Many [tutorials](https://easybuild.io/tutorial/) on EasyBuild are available
