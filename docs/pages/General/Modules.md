---
title: "Modules/en"
url: "https://docs.alliancecan.ca/wiki/Modules/en"
category: "General"
last_modified: "2022-06-22T01:35:02Z"
page_id: 17114
display_title: "Modules"
---

`<languages />`{=html}

In computing, a module is a unit of software that is designed to be independent, interchangeable, and contains everything necessary to provide the desired functionality. [^1] The term \"module\" may sometimes have a more specific meaning depending on the context. This page describes a few types of modules and suggests links to further documentation content.

## Disambiguation

### Lmod modules {#lmod_modules}

Also called \"environment modules\", Lmod modules are used to alter your (shell) environment so as to enable you to use a particular software package, or to use a non-default version of certain common software packages such as compilers. See [Using modules](https://docs.alliancecan.ca/Using_modules "Using modules"){.wikilink}.

### Python modules {#python_modules}

In Python, a module is a file of code (usually Python code) which can be loaded with the `import ...` or `from ... import ...` statements to provide functionality. A Python package is a collection of Python modules; the terms \"package\" and \"module\" are frequently interchanged in casual use. [^2]

Certain frequently used Python modules such as Numpy can be imported if you first load the `scipy-stack` Lmod module at the shell level. See [SciPy stack](https://docs.alliancecan.ca/Python#SciPy_stack "SciPy stack"){.wikilink} for details.

We maintain a large collection of [Python \"wheels.\"](https://docs.alliancecan.ca/Python#Available_wheels "Python "wheels.""){.wikilink} These are modules which are pre-compiled to be compatible with the [Standard software environments](https://docs.alliancecan.ca/Standard_software_environments "Standard software environments"){.wikilink}. Before importing modules from our wheels, you should create a [virtual environment](https://docs.alliancecan.ca/Python#Creating_and_using_a_virtual_environment "virtual environment"){.wikilink}.

Python modules which are not in the `scipy-stack` Lmod module or in our wheels collection can be installed from the internet as described in the [Installing packages](https://docs.alliancecan.ca/Python#Installing_packages "Installing packages"){.wikilink} section.

## Other related topics {#other_related_topics}

The main [Available software](https://docs.alliancecan.ca/Available_software "Available software"){.wikilink} page is a good starting point. Other related pages are:

- [Standard software environments](https://docs.alliancecan.ca/Standard_software_environments "Standard software environments"){.wikilink}: as of April 1, 2021, `StdEnv/2020` is the default collection of Lmod modules
- Lmod [modules specific to Niagara](https://docs.alliancecan.ca/modules_specific_to_Niagara "modules specific to Niagara"){.wikilink}
- Tables of Lmod modules optimized for [AVX](https://docs.alliancecan.ca/Modules_avx "AVX"){.wikilink}, **[AVX2](https://docs.alliancecan.ca/Modules_avx2 "AVX2"){.wikilink}** and **[AVX512](https://docs.alliancecan.ca/Modules_avx512 "AVX512"){.wikilink}** [CPU instructions](https://docs.alliancecan.ca/Standard_software_environments#Performance_improvements "CPU instructions"){.wikilink}
- [Category *Software*](https://docs.alliancecan.ca/:Category:Software "Category Software"){.wikilink}: a list of different software pages in this wiki, including commercial or licensed software

## Footnotes

[^1]: [Wikipedia, \"Modular programming\"](https://en.wikipedia.org/wiki/Modular_programming)

[^2]: [Tutorialspoint.com, \"What is the difference between a python module and a python package?\"](https://www.tutorialspoint.com/What-is-the-difference-between-a-python-module-and-a-python-package)
