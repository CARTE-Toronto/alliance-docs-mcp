---
title: "AI and Machine Learning/en"
url: "https://docs.alliancecan.ca/wiki/AI_and_Machine_Learning/en"
category: "General"
last_modified: "2024-10-16T16:55:18Z"
page_id: 11862
display_title: "AI and Machine Learning"
---

`<languages />`{=html}

To get the most out of our clusters for machine learning applications, special care must be taken. A cluster is a complicated beast that is very different from your local machine that you use for prototyping. Notably, a cluster uses a distributed filesystem, linking many storage devices seamlessly. Accessing a file on `/project` may `<i>`{=html}feel the same`</i>`{=html} as accessing one from the current node, but under the hood, these two IO operations have very different performance implications. In short, you need to [choose wisely where to put your data](https://docs.alliancecan.ca/#Managing_your_datasets "choose wisely where to put your data"){.wikilink}.

The sections below list links relevant to AI practitioners, and good practices to be observed on our clusters.

## Tutorials

A self-paced course on this topic is available from SHARCNET: [Introduction to Machine Learning](https://training.sharcnet.ca/courses/enrol/index.php?id=180).

If you are ready to port your program for using on one of our clusters, please take [our tutorial](https://docs.alliancecan.ca/Tutoriel_Apprentissage_machine/en "our tutorial"){.wikilink}.

A user-made tutorial showing all the steps necessary for setting up your local and Alliance environments for deep learning in Python is available [here](https://prashp.gitlab.io/post/compute-canada-tut/).

## Python

Python is very popular in the field of machine learning. If you (plan to) use it on our clusters, please refer to [our documentation about Python](https://docs.alliancecan.ca/Python "our documentation about Python"){.wikilink} to get important information about Python versions, virtual environments on login or on compute nodes, `multiprocessing`, Anaconda, Jupyter, etc.

### Avoid Anaconda {#avoid_anaconda}

We ask our users to avoid using Anaconda, and use virtualenv instead. Reasons are explained on the [Anaconda](https://docs.alliancecan.ca/Anaconda/en "Anaconda"){.wikilink} page.

`<b>`{=html}Switching to virtualenv is easy in most cases. Just install all the same packages, except CUDA, CuDNN and other low-level libraries, which are already installed on our clusters.`</b>`{=html}

## Useful information about software packages {#useful_information_about_software_packages}

Please refer to the page of your machine learning package of choice for useful information about how to install, common pitfalls, etc.:

- [TensorFlow](https://docs.alliancecan.ca/TensorFlow "TensorFlow"){.wikilink}
- [PyTorch](https://docs.alliancecan.ca/PyTorch "PyTorch"){.wikilink}
- [Keras](https://docs.alliancecan.ca/Keras "Keras"){.wikilink}
- [Torch](https://docs.alliancecan.ca/Torch "Torch"){.wikilink}
- [SpaCy](https://docs.alliancecan.ca/SpaCy "SpaCy"){.wikilink}
- [XGBoost](https://docs.alliancecan.ca/XGBoost "XGBoost"){.wikilink}
- [Scikit-Learn](https://docs.alliancecan.ca/Large_Scale_Machine_Learning_(Big_Data)#Scikit-Learn "Scikit-Learn"){.wikilink}
- [SnapML](https://docs.alliancecan.ca/Large_Scale_Machine_Learning_(Big_Data)#Snap_ML "SnapML"){.wikilink}

## Managing your datasets {#managing_your_datasets}

### Storage and file management {#storage_and_file_management}

Our clusters have a wide range of storage options to cover the needs of our very diverse users. These storage solutions range from high-speed temporary local storage to different kinds of long-term storage, so you can choose the storage medium that best corresponds to your needs and usage patterns. Please refer to our documentation on [Storage and file management](https://docs.alliancecan.ca/Storage_and_file_management "Storage and file management"){.wikilink}.

### Choosing the right storage type for your dataset {#choosing_the_right_storage_type_for_your_dataset}

- If your dataset is around 10 GB or less, it can probably fit in the memory, depending on how much memory your job has. You should not read data from disk during your machine learning tasks.
- If your dataset is around 100 GB or less, it can fit in the local storage of the compute node; please transfer it there at the beginning of the job. This storage is orders of magnitude faster and more reliable than shared storage (home, project, scratch). A temporary directory is available for each job at \$SLURM_TMPDIR. An example is given in [our tutorial](https://docs.alliancecan.ca/Tutoriel_Apprentissage_machine/en "our tutorial"){.wikilink}. A caveat of local node storage is that a job from another user might be using it fully, leaving you no space (we are currently studying this problem). However, you might also get lucky and have a whole terabyte at your disposal.
- If your dataset is larger, you may have to leave it in the shared storage. You can leave your datasets permanently in your project space. Scratch space can be faster, but it is not for permanent storage. Also, all shared storage (home, project, scratch) is for storing and reading at low frequencies (e.g. 1 large chunk every 10 seconds, rather than 10 small chunks every second).

### Datasets containing lots of small files (e.g. image datasets) {#datasets_containing_lots_of_small_files_e.g._image_datasets}

In machine learning, it is common to have to manage very large collections of files, meaning hundreds of thousands or more. The individual files may be fairly small, e.g. less than a few hundred kilobytes. In these cases, problems arise:

- filesystem [quotas](https://docs.alliancecan.ca/Storage_and_file_management#Filesystem_quotas_and_policies "quotas"){.wikilink} on our clusters limit the number of filesystem objects;
- your software could be significantly slowed down from streaming lots of small files from `/project` (or `/scratch`) to a compute node.

On a distributed filesystem, data should be stored in large single-file archives. On this subject, please refer to [Handling large collections of files](https://docs.alliancecan.ca/Handling_large_collections_of_files "Handling large collections of files"){.wikilink}.

## Long running computations {#long_running_computations}

If your computations are long, you should use checkpointing. For example, if your training time is 3 days, you should split it in 3 chunks of 24 hours. This will prevent you from losing all the work in case of an outage, and give you an edge in terms of priority (more nodes are available for short jobs). Most machine learning libraries natively support checkpointing; the typical case is covered in our [tutorial](https://docs.alliancecan.ca/Tutoriel_Apprentissage_machine/en#Checkpointing_a_long-running_job "tutorial"){.wikilink}. If your program does not natively support this, we provide a [general checkpointing solution](https://docs.alliancecan.ca/Points_de_contrôle/en "general checkpointing solution"){.wikilink}.

For more examples, please see

[Checkpointing with PyTorch](https://docs.alliancecan.ca/PyTorch#Creating_model_checkpoints "Checkpointing with PyTorch"){.wikilink}

[Checkpointing with TensorFlow](https://docs.alliancecan.ca/TensorFlow#Creating_model_checkpoints "Checkpointing with TensorFlow"){.wikilink}

## Running many similar jobs {#running_many_similar_jobs}

If you are in one of these situations:

- Hyperparameter search
- Training many variants of the same method
- Running many optimization processes of similar duration

\... you should consider grouping many jobs into one. [META](https://docs.alliancecan.ca/META:_A_package_for_job_farming "META"){.wikilink}, [GLOST](https://docs.alliancecan.ca/GLOST "GLOST"){.wikilink}, and [GNU Parallel](https://docs.alliancecan.ca/GNU_Parallel "GNU Parallel"){.wikilink} are available to help you with this.

## Experiment tracking and hyperparameter optimization {#experiment_tracking_and_hyperparameter_optimization}

[Weights & Biases (wandb)](https://docs.alliancecan.ca/Weights_&_Biases_(wandb) "Weights & Biases (wandb)"){.wikilink} and [Comet.ml](https://docs.alliancecan.ca/Comet.ml "Comet.ml"){.wikilink} can help you get the most out of your compute allocation, by

- allowing easier tracking and analysis of training runs;
- providing Bayesian hyperparameter search.

Note that Comet and Wandb are not currently available on Graham.

## Large-scale machine learning (big data) {#large_scale_machine_learning_big_data}

Modern deep learning packages like Pytorch and TensorFlow include utilities to handle large-scale training natively and tutorials on how to do it abound. Scaling classic machine learning (i.e., not deep learning) methods, however, is not as widely discussed and can often be a frustrating problem to solve. [This guide](https://docs.alliancecan.ca/Large_Scale_Machine_Learning_(Big_Data) "This guide"){.wikilink} contains ideas and practical options, along with tutorials, to tackle training classic ML models on very large datasets.

## Troubleshooting

### Determinism with RNN using CUDA {#determinism_with_rnn_using_cuda}

RNN and multi-head attention API calls may exhibit non-deterministic behaviour when the cuDNN library is built with CUDA Toolkit 10.2 or higher. The user can eliminate the non-deterministic behaviour of cuDNN RNN and multi-head attention APIs by setting a single buffer size in the CUBLAS_WORKSPACE_CONFIG environmental variable, for example, :16:8 or :4096:2, which instructs cuBLAS to allocate eight buffers of 16 KB each in GPU memory or two buffers of 4 MB each.
