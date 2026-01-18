---
title: "MLflow"
url: "https://docs.alliancecan.ca/wiki/MLflow"
category: "General"
last_modified: "2026-01-14T15:02:43Z"
page_id: 32184
display_title: "MLflow"
---

MLflow is an open-source experiment tracking and observability platform for AI and Machine Learning development. Contrary to to other similar platforms such as WandB and Comet.ml, it is possible to run MLflow locally on our systems and track experiments without internet connection and without any data ever leaving the cluster.

== On JupyterHub ==

On clusters where JupyterHub is available, you can launch MLflow by clicking on the following icon on an active launcher tab:

This will open the application on a new tab on your web browser. Switch to that tab to start using MLflow.

Upon launching MLflow on JupyterHub, a new file named mlruns.db will be created inside your current directory. This file is a Sqlite database where all metrics being tracked by MLflow will be stored. If you also use MLflow to store and manage trained models, checkpoints and other artifacts, these will be saved to a new automatically created directory named mlruns, also in the current directory. Deleting mlruns.db and/or mlruns will result in loss of these saved metrics and artifacts and they will no longer be available on the MLFlow tab .

Note that, since MLFlow will be using a Sqlite database as a backend to store metrics, it is not recommended to track more than one experiment at a time when using MLFlow on JupyterHub. Sqlite is not designed to have multiple processes write to the same database at the same time safely - attempting to do so may result in deadlocks and lead to your code hanging until the end of your session.

To track an experiment in real time on JupyterHub, first you will need to install the mlflow package. To do this, load the Arrow module by clicking on the modules menu (the cube-looking icon on the left-hand side menu), searching for the name "arrow", selecting the desired version of the module and clicking on "load".

Next, you can install the mlflow package directly in a cell on your notebook with pip install --no-index mlflow. For an example on how to use the mlflow packages to send training metrics to the MLFlow app, see MLFlow's official documentation.