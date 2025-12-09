---
title: "Using Conda in Apptainer/en"
url: "https://docs.alliancecan.ca/wiki/Using_Conda_in_Apptainer/en"
category: "General"
last_modified: "2025-12-03T14:48:20Z"
page_id: 32061
display_title: "Using Conda in Apptainer"
---

`<noinclude>`{=html} `<languages />`{=html} `</noinclude>`{=html}

We will preface this tutorial on how to use Conda inside a container with the following **important notes**:

- Even inside a container, Conda should not be your preferred solution. Priority should always be given to using [modules](https://docs.alliancecan.ca/Modules "modules"){.wikilink} from our [software stack](https://docs.alliancecan.ca/Available_software "software stack"){.wikilink}, and [wheels](https://docs.alliancecan.ca/Python "wheels"){.wikilink} from our [Python wheelhouse](https://docs.alliancecan.ca/Available_Python_wheels "Python wheelhouse"){.wikilink}. These are optimized for our systems and we are better equipped to provide support if you use them. Please [contact us](https://docs.alliancecan.ca/Technical_support "contact us"){.wikilink} if you need a module or a Python package that is not currently available on our systems.
- This tutorial will use the [micromamba](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html) package manager instead of Conda. If you choose to use Conda instead, keep in mind that its use is subject to [Anaconda\'s Terms of Service](https://legal.anaconda.com/policies/en?name=terms-of-service#terms-of-service) and might require a [commercial license](https://www.anaconda.com/pricing/terms-of-service-faqs).
- This tutorial shows how to create a read-only image, i.e., a one-off `.sif` file containing a Conda environment that has everything you need to run your application. We strongly discourage installing software interactively with Conda inside a container and will not show how to do this here.

Creating an Apptainer image and using Conda to install software inside it is a 3-step process. The first step is to create a `.yml` file describing the Conda environment we wish to create inside the container. In the example that follows, we create the file `environment.yml` . This file is where we give our environment a name, then give Conda a list of packages that must be installed and the channels where to look for them. For more information [see here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#create-env-file-manually).

Second, we create an Apptainer [image definition file](https://apptainer.org/docs/user/main/definition_files.html). This file, here called `image.def`, describes what are the steps Apptainer should take to create our image. These steps are:

1.  Pull a Docker image from DockerHub that has the micromamba package manager pre-installed.
2.  Create a copy of the Conda environment definition file `environment.yml` inside the container
3.  Call micromamba and have it configure the environment defined in `environment.yml`.

The last step is to build the Apptainer image using the definition file above:

`  module load apptainer`\
`  APPTAINER_BIND=' ' apptainer build image.sif image.def`

You can test that your image provides `multiqc`, for example, like this:
