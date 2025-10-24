---
title: "Faiss/en"
url: "https://docs.alliancecan.ca/wiki/Faiss/en"
category: "General"
last_modified: "2024-05-02T20:11:23Z"
page_id: 25426
display_title: "Faiss"
---

`<languages />`{=html}

[Faiss](https://github.com/facebookresearch/faiss/wiki) is a library for efficient similarity search and clustering of dense vectors. It contains algorithms that search in sets of vectors of any size, up to ones that possibly do not fit in RAM. It also contains supporting code for evaluation and parameter tuning. Faiss is written in C++ with complete wrappers for Python (versions 2 and 3). Some of the most useful algorithms are implemented on GPU. It is developed primarily at [Meta AI Research](https://research.facebook.com/) with help from external contributors.

\_\_TOC\_\_

## Python bindings {#python_bindings}

The module contains bindings for multiple Python versions. To discover which are the compatible Python versions, run

Or search directly `<i>`{=html}faiss-cpu`</i>`{=html}, by running

where `X.Y.Z` represent the desired version.

### Usage

1\. Load the required modules.

where `X.Y.Z` represent the desired version.

2\. Import Faiss.

If the command displays nothing, the import was successful.

#### Available Python packages {#available_python_packages}

Other Python packages depend on `faiss-cpu` or `faiss-gpu` bindings in order to be installed. The `faiss` module provides:

- `faiss`
- `faiss-gpu`
- `faiss-cpu`

`fgrep faiss`

\|result= faiss-gpu 1.7.4 faiss-cpu 1.7.4 faiss 1.7.4 }}

With the `faiss` module loaded, package dependency for the above extensions will be satisfied.
