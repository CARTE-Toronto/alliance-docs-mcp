---
title: "SpaCy/en"
url: "https://docs.alliancecan.ca/wiki/SpaCy/en"
category: "General"
last_modified: "2022-07-19T22:44:03Z"
page_id: 9301
display_title: "SpaCy"
---

`<languages />`{=html}

[spaCy](https://spacy.io/) is a Python package that provides industrial-strength natural language processing.

# Installation

## Latest available wheels {#latest_available_wheels}

To see the latest version of `spaCy` that we have built:

For more information on listing wheels, see [ listing available wheels](https://docs.alliancecan.ca/Python#Listing_available_wheels " listing available wheels"){.wikilink}.

## Pre-build {#pre_build}

The preferred option is to install it using the python [wheel](https://pythonwheels.com/) that we compile, as follows:

:   1\. Load python 3.6 module: `python/3.6`
:   2\. Create and activate a [virtual environment](https://docs.alliancecan.ca/Python#Creating_and_using_a_virtual_environment "virtual environment"){.wikilink}.
:   3\. Install `spaCy` in the virtual environment with `pip install`. For both GPU and CPU support:
:   
:   If you only need CPU support:
:   

**GPU version**: At the present time, in order to use the GPU version you need to add the CUDA libraries to `LD_LIBRARY_PATH`: \$CUDA_HOME/lib64:\$LD_LIBRARY_PATH }}

If you want to use the [Pytorch](https://docs.computecanada.ca/wiki/PyTorch) wrapper with `thinc`, you\'ll also need to install the `torch_cpu` or `torch_gpu` wheel.
