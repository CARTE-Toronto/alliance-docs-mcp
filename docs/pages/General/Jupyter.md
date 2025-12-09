---
title: "Jupyter/en"
url: "https://docs.alliancecan.ca/wiki/Jupyter/en"
category: "General"
last_modified: "2022-07-19T12:51:41Z"
page_id: 18540
display_title: "Jupyter"
---

`<languages />`{=html}

# The Jupyter vocabulary and related wiki pages {#the_jupyter_vocabulary_and_related_wiki_pages}

- **Jupyter**: an implementation of Web applications and notebook rendering
  - *Google Colab* would be another implementation of the same kind of environment
- **Jupyter Application**: like a regular application, but is displayed in a separate Web browser tab. The application has access to the data stored remotely on the server, and the heavy computations are also handled by the remote server
- [**JupyterHub**: a Web server hosting Jupyter portals and kernels](https://docs.alliancecan.ca/JupyterHub "JupyterHub: a Web server hosting Jupyter portals and kernels"){.wikilink}

## JupyterLab

A Web portal with a modern interface for managing and running applications, as well as rendering notebook files of various kernels. For more details:

- [**JupyterLab via JupyterHub**: a pre-installed JupyterLab environment](https://docs.alliancecan.ca/JupyterHub#JupyterLab "JupyterLab via JupyterHub: a pre-installed JupyterLab environment"){.wikilink}, with a default Python kernel and the access to software modules
- [**JupyterLab from a virtual environment**: a self-made environment](https://docs.alliancecan.ca/Advanced_Jupyter_configuration "JupyterLab from a virtual environment: a self-made environment"){.wikilink} to be launched by a Slurm job

## Jupyter Notebook {#jupyter_notebook}

An older Web portal for managing and running applications, as well as rendering notebook files of various kernels. For more details:

- [**Jupyter Notebook via JupyterHub**: a pre-installed Jupyter Notebook environment](https://docs.alliancecan.ca/JupyterHub#User_Interface "Jupyter Notebook via JupyterHub: a pre-installed Jupyter Notebook environment"){.wikilink}, with a default Python kernel and the access to software modules
- [**Jupyter Notebook from a virtual environment**: a self-made environment](https://docs.alliancecan.ca/JupyterNotebook "Jupyter Notebook from a virtual environment: a self-made environment"){.wikilink} to be launched by a Slurm job

## Kernel

The active service behind the Web interface. There are:

- Notebook kernels (e.g. Python, R, Julia)
- Application kernels (e.g. RStudio, VSCode)

## Notebook

A page of executable cells of code and formatted text:

- **IPython notebooks**: a notebook executed by a Python kernel, and has some IPython interactive special commands that are not supported by a regular Python shell
