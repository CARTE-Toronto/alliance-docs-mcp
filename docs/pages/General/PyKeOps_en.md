---
title: "PyKeOps/en"
url: "https://docs.alliancecan.ca/wiki/PyKeOps/en"
category: "General"
last_modified: "2024-09-30T19:02:03Z"
page_id: 26510
display_title: "PyKeOps"
---

`<languages />`{=html} \_\_FORCETOC\_\_ The [KeOps](https://www.kernel-operations.io/keops/index.html) library lets you compute reductions of large arrays whose entries are given by a mathematical formula or a neural network. It combines efficient C++ routines with an automatic differentiation engine and can be used with [Python](https://docs.alliancecan.ca/Python "Python"){.wikilink} ([NumPy](https://numpy.org/doc/stable/), [PyTorch](https://docs.alliancecan.ca/PyTorch "PyTorch"){.wikilink}), [MATLAB](https://docs.alliancecan.ca/MATLAB "MATLAB"){.wikilink} and [R](https://docs.alliancecan.ca/R "R"){.wikilink}.

# Available versions {#available_versions}

PyKeOps is available on our clusters as prebuilt Python packages (wheels). You can list available versions with `avail_wheels`.

# Installing PyKeOps in a Python virtual environment {#installing_pykeops_in_a_python_virtual_environment}

1\. Load runtime dependencies.

2\. Create and activate a [ Python virtual environment](https://docs.alliancecan.ca/Python#Creating_and_using_a_virtual_environment " Python virtual environment"){.wikilink}.

3\. Install a specific version of PyKeOps and its Python dependencies. X.Y.Z }} where `X.Y.Z` is the exact desired version, for instance `2.2.3`. You can omit to specify the version in order to install the latest one available from the wheelhouse.

4\. Validate it.

5\. Freeze the environment and requirements set.

6\. Remove the local virtual environment.

# Running KeOps {#running_keops}

You can run PyKeOps on CPU or GPU.

1\. Write your job submission script. `<tabs>`{=html} `<tab name="CPU">`{=html}

`</tab>`{=html} `<tab name="GPU">`{=html}

`</tab>`{=html} `</tabs>`{=html}

2\. Before submitting your job, it is important to test that your submission script will start without errors. You can do a quick test in an [interactive job](https://docs.alliancecan.ca/Running_jobs#Interactive_jobs "interactive job"){.wikilink}.

3\. Submit your job to the scheduler.
