---
title: "AlphaFold3/en"
url: "https://docs.alliancecan.ca/wiki/AlphaFold3/en"
category: "General"
last_modified: "2025-07-18T17:48:59Z"
page_id: 27116
display_title: "AlphaFold3"
---

`<languages />`{=html}

This page discusses how to use AlphaFold v3.0.

Source code and documentation for AlphaFold3 can be found at their [GitHub page](https://github.com/google-deepmind/alphafold3). Any publication that discloses findings arising from use of this source code or the model parameters should [cite](https://github.com/google-deepmind/alphafold3#citing-this-work) the [AlphaFold3 paper](https://doi.org/10.1038/s41586-024-07487-w).

## Available versions {#available_versions}

AlphaFold3 is available on our clusters as prebuilt Python packages (wheels). You can list available versions with `avail_wheels`.

AlphaFold2 is still available. Documentation is [here](https://docs.alliancecan.ca/AlphaFold2 "here"){.wikilink}.

## Creating a requirements file for AlphaFold3 {#creating_a_requirements_file_for_alphafold3}

1\. Load AlphaFold3 dependencies.

2\. Download run script. `<tabs>`{=html} `<tab name="3.0.1">`{=html}

`</tab>`{=html} `<tab name="3.0.0">`{=html}

`</tab>`{=html} `</tabs>`{=html}

3\. Create and activate a Python virtual environment.

4\. Install a specific version of AlphaFold3 and its Python dependencies. X.Y.Z }} where `X.Y.Z` is the exact desired version, for instance `3.0.0`. You can omit to specify the version in order to install the latest one available from the wheelhouse.

5\. Build data.

This will create data files inside your virtual environment.

6\. Validate it.

7\. Freeze the environment and requirements set.

8\. Deactivate the environment.

9\. Clean up and remove the virtual environment.

The virtual environment will be created in your job instead.

## Model

You can obtain the model by requesting it from Google. They aim to respond to requests within 2-3 business days. Please see [Obtaining Model Parameters](https://github.com/google-deepmind/alphafold3?tab=readme-ov-file).

## Databases

Note that AlphaFold3 requires a set of databases.

`<b>`{=html}Important:`</b>`{=html} The databases must live in the `$SCRATCH` directory.

1\. Download the fetch script

2\. Download the databases

## Running AlphaFold3 in stages {#running_alphafold3_in_stages}

Alphafold3 must be run in [stages](https://github.com/google-deepmind/alphafold3/blob/main/docs/performance.md#running-the-pipeline-in-stages), that is:

1.  Splitting the CPU-only data pipeline from model inference (which requires a GPU), to optimise cost and resource usage.
2.  Caching the results of MSA/template search, then reusing the augmented JSON for multiple different inferences across seeds or across variations of other features (e.g. a ligand).

For reference on Alphafold3:

- see [inputs](https://github.com/google-deepmind/alphafold3/blob/main/docs/input.md)
- see [outputs](https://github.com/google-deepmind/alphafold3/blob/main/docs/output.md)
- see [performance](https://github.com/google-deepmind/alphafold3/blob/main/docs/performance.md)

### 1. Data pipeline (CPU) {#data_pipeline_cpu}

Edit the following submission script according to your needs.

### 2. Model inference {#model_inference}

Edit the following submission script according to your needs.

### 3. Job submission {#job_submission}

Then, submit the jobs to the scheduler.

#### Independent jobs {#independent_jobs}

Wait until it complete, then submit the second stage:

#### Dependent jobs {#dependent_jobs}

\$(sbatch alphafold3-data.sh) \|jid2\$(sbatch \--dependencyafterok:\$jid1 alphafold3-inference.sh) \|sq }} If the first stage fails, you will have to manually cancel the second stage:

## Troubleshooting

### Out of memory (GPU) {#out_of_memory_gpu}

If you would like to run AlphaFold3 on inputs larger than 5,120 tokens, or on a GPU with less memory (an A100 with 40 GB of memory, for instance), you can enable [unified memory](https://github.com/google-deepmind/alphafold3/blob/main/docs/performance.md#unified-memory)

In your submission script for the inference stage, add these environment variables:

``` bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_FORCE_UNIFIED_MEMORY=true
export XLA_CLIENT_MEM_FRACTION=2.0  # 2 x 40GB = 80 GB
```

and adjust the amount of memory allocated to your job accordingly, for instance: `#SBATCH --mem=80G`
