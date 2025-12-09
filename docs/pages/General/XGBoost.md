---
title: "XGBoost/en"
url: "https://docs.alliancecan.ca/wiki/XGBoost/en"
category: "General"
last_modified: "2024-06-10T16:15:07Z"
page_id: 9832
display_title: "XGBoost"
---

`<languages />`{=html}

**[XGBoost](https://xgboost.readthedocs.io/en/latest/)** *is an optimized distributed gradient boosting library designed to be highly efficient, flexible and portable*. It is a popular package used for a wide variety of machine learning and datascience tasks, serving the role of a convenient, domain-agnostic black box classifier. XGBoost provides GPU accelerated learning for some problems, and Compute Canada provides a GPU enabled build.

For detailed documentation on using the library, please consult the [xgboost documentation](https://xgboost.readthedocs.io/en/latest/get_started.html). There is a [separate section for GPU-enabled training](https://xgboost.readthedocs.io/en/latest/gpu/index.html).

## Python Module Installation {#python_module_installation}

A very common way to use XGBoost is though its python interface, provided as the `xgboost` python module. Compute Canada provides an optimized, multi-GPU enabled build as a [Python](https://docs.alliancecan.ca/Python "Python"){.wikilink} wheel; readers can should familiarize themselves with the use of [ Python virtual environments](https://docs.alliancecan.ca/Python#Creating_and_using_a_virtual_environment " Python virtual environments"){.wikilink} before starting an XGBoost project.

Currently, version 0.81 of XGBoost is available. The following commands illustrate the needed package and module: 0.81 \--no-index }}
