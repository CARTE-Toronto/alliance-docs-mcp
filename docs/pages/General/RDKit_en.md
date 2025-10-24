---
title: "RDKit/en"
url: "https://docs.alliancecan.ca/wiki/RDKit/en"
category: "General"
last_modified: "2025-01-27T21:11:16Z"
page_id: 27750
display_title: "RDKit"
---

`<languages />`{=html}

[RDKit](https://www.rdkit.org/) is a collection of cheminformatics and machine-learning software written in C++ and Python.

\_\_FORCETOC\_\_

# Available versions {#available_versions}

`rdkit` C++ libraries and Python bindings are available as a module.

You can find available versions with:

and look for more information on a specific version with:

where `X.Y.Z` is the exact desired version, for instance `2024.03.5`.

# Python bindings {#python_bindings}

The module contains bindings for multiple Python versions. To discover which are the compatible Python versions, run:

where `X.Y.Z` represents the desired version.

## rdkit as a Python package dependency {#rdkit_as_a_python_package_dependency}

When `rdkit` is a dependency of another package, the dependency needs to be fulfilled:

1\. Deactivate any Python virtual environment.

`<b>`{=html}Note:`</b>`{=html} If you had a virtual environment activated, it is important to deactivate it first, then load the module, before reactivating your virtual environment.

2\. Load the module.

3\. Check that it is visible by `pip`

`grep rdkit`

\|result= rdkit 2024.3.5 }}

If no errors are raised, then everything is OK!

4\. [Create a virtual environment and install your packages](https://docs.alliancecan.ca/Python#Creating_and_using_a_virtual_environment "Create a virtual environment and install your packages"){.wikilink}.

# Troubleshooting

## ModuleNotFoundError: No module named \'rdkit\' {#modulenotfounderror_no_module_named_rdkit}

If `rdkit` is not accessible, you may get the following error when importing it: `ModuleNotFoundError: No module named 'rdkit'`

Possible solutions:

- check which Python versions are compatible with your loaded RDKit module using `module spider rdkit/X.Y.Z`. Once a compatible Python module is loaded, check that `python -c 'import rdkit'` works.
- load the module before activating your virtual environment: please see the [rdkit as a package dependency](https://docs.alliancecan.ca/RDKit#rdkit_as_a_Python_package_dependency "rdkit as a package dependency"){.wikilink} section above.

See also [ModuleNotFoundError: No module named \'X\'](https://docs.alliancecan.ca/Python#ModuleNotFoundError:_No_module_named_'X' "ModuleNotFoundError: No module named 'X'"){.wikilink}.
