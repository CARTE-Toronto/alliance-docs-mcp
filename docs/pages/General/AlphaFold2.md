---
title: "AlphaFold2/en"
url: "https://docs.alliancecan.ca/wiki/AlphaFold2/en"
category: "General"
last_modified: "2024-11-22T13:54:29Z"
page_id: 20387
display_title: "AlphaFold2"
---

AlphaFold
is a machine learning model for the prediction of protein folding.

This page discusses how to use AlphaFold v2.0, the version that was entered in CASP14 and published in Nature.

Source code and documentation for AlphaFold can be found at their GitHub page.
Any publication that discloses findings arising from use of this source code or the model parameters should cite the AlphaFold paper.

== Available versions ==
AlphaFold is available on our clusters as prebuilt Python packages (wheels). You can list available versions with avail_wheels.

== Installing AlphaFold in a Python virtual environment ==

1. Load AlphaFold dependencies.

As of July 2022, only Python 3.7 and 3.8 are supported.

2. Create and activate a Python virtual environment.

3. Install a specific version of AlphaFold and its Python dependencies.
X.Y.Z
}}
where X.Y.Z is the exact desired version, for instance 2.2.4.
You can omit to specify the version in order to install the latest one available from the wheelhouse.

4. Validate it.

5. Freeze the environment and requirements set.

== Databases ==
Note that AlphaFold requires a set of databases.

The databases are available in
/cvmfs/bio.data.computecanada.ca/content/databases/Core/alphafold2_dbs/.

AlphaFold databases on CVMFS undergo yearly updates. In January 2024, the database was updated and is accessible in folder 2024_01.
/cvmfs/bio.data.computecanada.ca/content/databases/Core/alphafold2_dbs/2024_01/
}}

You can also choose to download the databases locally into your $SCRATCH directory.

Important: The databases must live in the $SCRATCH directory.

1. From a DTN or login node, create the data folder.
$SCRATCH/alphafold/data
|mkdir -p $DOWNLOAD_DIR
}}

2. With your modules loaded and virtual environment activated, you can download the data.

Note that this step cannot be done from a compute node. It should be done on a data transfer node (DTN) on clusters that have them (see Transferring data). On clusters that have no DTN, use a login node instead. Since the download can take up to a full day, we suggest using a terminal multiplexer. You may encounter a Client_loop: send disconnect: Broken pipe error message. See Troubleshooting below.

1. Set DOWNLOAD_DIR.
/datashare/alphafold
}}

Afterwards, the structure of your data should be similar to

== Running AlphaFold ==

Edit one of following submission scripts according to your needs.

Then, submit the job to the scheduler.

== Troubleshooting ==
=== Broken pipe error message ===
When downloading the database, you may encounter a Client_loop: send disconnect: Broken pipe error message. It is hard to find the exact cause for this error message. It could be as simple as an unusually high number of users working on the login node, leaving less space for you to upload data.

*One solution is to use a terminal multiplexer. Note that you could still encounter this error message but less are the chances.

*A second solution is to use the database that is already present on the cluster. /cvmfs/bio.data.computecanada.ca/content/databases/Core/alphafold2_dbs/2023_07/.

*Another option is to download the full database in sections. To have access to the different download scripts, after loading the module and activated your virtual environment, you simply enter download_ in your terminal and tap twice on the tab keyboard key to visualize all the scripts that are available. You can manually download sections of the database by using the available script, as for instance download_pdb.sh.