---
title: "Huggingface"
url: "https://docs.alliancecan.ca/wiki/Huggingface"
category: "General"
last_modified: "2025-08-01T18:36:50Z"
page_id: 23413
display_title: "Huggingface"
---

[Hugging Face](http://huggingface.co) is an organization that builds and maintains several popular open-source software packages widely used in Artificial Intelligence research. In this article, you will find information and tutorials on how to use packages from the Hugging Face ecosystem on our clusters.

# Transformers

Transformers is a python package that provides APIs and tools to easily download and train state-of-the-art models, pre-trained on various tasks in multiple domains.

## Installing Transformers {#installing_transformers}

Our recommendation is to install it using our provided Python [wheel](https://pythonwheels.com/) as follows:

:   1\. Load a Python [module](https://docs.alliancecan.ca/Utiliser_des_modules/en#Sub-command_load "module"){.wikilink}, thus `module load python`
:   2\. Create and start a [virtual environment](https://docs.alliancecan.ca/Python#Creating_and_using_a_virtual_environment "virtual environment"){.wikilink}.
:   3\. Install Transformers in the virtual environment with `pip install`.

<!-- -->

:   

## Downloading pre-trained models {#downloading_pre_trained_models}

To download a pre-trained model from the Hugging Face model hub, choose one of the options below and follow the instructions on the **login node** of the cluster you are working on. Models must be downloaded on a login node to avoid idle compute while waiting for resources to download.

### Using git lfs {#using_git_lfs}

Pre-trained models are usually made up of fairly large binary files. The Hugging Face makes these files available for download via [Git Large File Storage](https://git-lfs.com/). To download a model, load the `git-lfs` module and clone your chosen model repository from the model hub:

`module load git-lfs/3.4.0`\
`git clone --depth 1 --jobs 1 `[`https://huggingface.co/bert-base-uncased`](https://huggingface.co/bert-base-uncased)

Now that you have a copy of the pre-trained model saved locally in the cluster\'s filesystem, you can load it with a python script inside a job with the `local_files_only` option to avoid attempts to download it from the web:

`from transformers import AutoModel, AutoTokenizer`\
`model = AutoModel.from_pretrained("/path/to/where/you/cloned/the/model", local_files_only=True)`\
`tokenizer = AutoTokenizer.from_pretrained("/path/to/where/you/cloned/the/model", local_files_only=True)`

### Using the Hugging Face command line interface {#using_the_hugging_face_command_line_interface}

The `huggingface_hub` package contains a command line interface (CLI) which can be used to download models. For example, to download the model `Zephyr-7b-beta`, first install `huggingface_hub` in a virtual environment, then **on a login node** run:

` HF_HUB_DISABLE_XET=1 hf download --max-workers=1 HuggingFaceH4/zephyr-7b-beta`

Note that we set the variable `HF_HUB_DISABLE_XET` to avoid using the `hf_xet` package to download models. This package, meant to make downloading artifacts from the Hugging Face more efficient, currently leads to failures on our systems and should not be used at this time.

### Using python {#using_python}

It is also possible to download pre-trained models using Python instead of Git. The following must be executed **on a login node** as an internet connection is required to download the model files:

`from transformers import AutoModel, AutoTokenizer`\
`model = AutoModel.from_pretrained("bert-base-uncased")`\
`tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")`

This will store the pre-trained model files in a cache directory, which defaults to `$HOME/.cache/huggingface/hub`. You can change the cache directory by setting the environment variable `TRANSFORMERS_CACHE` **before** you import anything from the transformers package in your Python script. For example, the following will store model files in the current working directory:

`import os`\
`os.environ['TRANSFORMERS_CACHE']="./"`\
`from transformers import AutoModel, AutoTokenizer`\
`model = AutoModel.from_pretrained("bert-base-uncased")`\
`tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")`

Whether you change the default cache directory location or not, you can load the pre-trained model from disk in a job by using the `local_files_only` option:

`from transformers import AutoModel, AutoTokenizer`\
`model = AutoModel.from_pretrained("/path/to/where/model/is/saved", local_files_only=True)`\
`tokenizer = AutoTokenizer.from_pretrained("/path/to/where/model/is/saved", local_files_only=True)`

### Using a pipeline {#using_a_pipeline}

Another frequently used way of loading a pre-trained model is via a [pipeline](https://huggingface.co/docs/transformers/main_classes/pipelines). **On a login node** you can simply pass a model name or a type of task as an argument to `pipeline`. This will download and store the model at the default cache location:

`from transformers import pipeline`\
`pipe = pipeline("text-classification")`

In an environment without internet connection however, such as inside a job, you must specify the location of the model as well as its tokenizer when calling `pipeline`:

` from transformers import pipeline, AutoModel, AutoTokenizer`\
` model = AutoModel.from_pretrained("/path/to/where/model/is/saved", local_files_only=True)`\
` tokenizer = AutoTokenizer.from_pretrained("/path/to/where/model/is/saved", local_files_only=True)`\
` pipe = pipeline(task = "text-classification", model = model, tokenizer = tokenizer)`

Failing to do so will result in `pipeline` attempting to download models from the internet, which will result in a connection timeout error during a job.

# Datasets

Datasets is a python package for easily accessing and sharing datasets for Audio, Computer Vision, and Natural Language Processing (NLP) tasks.

## Installing Datasets {#installing_datasets}

Our recommendation is to install it using our provided Python [wheel](https://pythonwheels.com/) as follows:

:   1\. Load a Python [module](https://docs.alliancecan.ca/Utiliser_des_modules/en#Sub-command_load "module"){.wikilink}, thus `module load python`
:   2\. Create and start a [virtual environment](https://docs.alliancecan.ca/Python#Creating_and_using_a_virtual_environment "virtual environment"){.wikilink}.
:   3\. Load the Arrow module. This will make the `pyarrow` package (a dependency of Datasets) available inside your virtualenv.
:   3\. Install Datasets in the virtual environment with `pip install`.

<!-- -->

:   

:   

**Note:** you will need to load the arrow module you every time intend to import the Datasets package in your Python script.

## Downloading Datasets {#downloading_datasets}

The exact method to download and use a dataset from the Hugging Face hub depends on a number of factors such as format and the type of task for which the data will be used. Regardless of the exact method used, any download must be performed **on a login node**. See [the package\'s official documentation](https://huggingface.co/docs/datasets/loading) for details on how to download different types of dataset.

Once the dataset has been downloaded, it will be stored locally in a cache directory, which defaults to `$HOME/.cache/huggingface/datasets`. It is possible to change the default cache location by setting the environment variable `HF_DATASETS_CACHE` **before** you import anything from the Datasets package in your python script.

To load a dataset in a job where there is no internet connection, set the environment variable `HF_DATASETS_OFFLINE=1` and specify the location of the cache directory where the dataset is stored when calling `load_dataset()`:

`import os`\
`os.environ['HF_DATASETS_OFFLINE'] = '1'`\
`from datasets import load_dataset`\
`dataset = load_dataset("/path/to/loading_script/of/the/dataset")`

# Evaluate

Evaluate is a library for easily evaluating machine learning models and datasets.

With a single line of code, you get access to dozens of evaluation methods for different domains (NLP, Computer Vision, Reinforcement Learning, and more.)

## Installing Evaluate {#installing_evaluate}

Our recommendation is to install it using our provided Python [wheel](https://pythonwheels.com/) as follows:

:   1\. Load a Python [module](https://docs.alliancecan.ca/Utiliser_des_modules/en#Sub-command_load "module"){.wikilink}, thus `module load python`
:   2\. Create and start a [virtual environment](https://docs.alliancecan.ca/Python#Creating_and_using_a_virtual_environment "virtual environment"){.wikilink}.
:   3\. Load the Arrow module. This will make the `pyarrow` package (a dependency of Evaluate) available inside your virtualenv.
:   3\. Install Evaluate in the virtual environment with `pip install`.

<!-- -->

:   

:   

**Note:** you will need to load the arrow module you every time intend to import the Evaluate package in your Python script.

## Downloading Evaluators {#downloading_evaluators}

The default behaviour of this package when loading an evaluator is to attempt to download it from the internet. As such, you must first download any evaluators you wish to use in your code **on a login node, before submitting a job.**

To download an evaluator, simply call the `evaluate.load()` method. For example, to download an accuracy evaluator, run the following **on a login node:**

`  import evaluate`\
`  evaluate.load("accuracy")`

\'\'\'

Inside a job on compute nodes **without internet connection**, you must set the environment variable `HF_EVALUATE_OFFLINE=1` to prevent Evaluate from attempting to download evaluators from the web.

Note that evaluators are saved at the default location `$HOME/.cache/huggingface/evaluate`. You can change this by pointing the environment variable `HF_HOME` to your desired storage location. Note that setting this variable will change the storage location for all Hugging Face ecosystem libraries such as Transformers and Datasets.

# Accelerate

Accelerate is a package that enables any PyTorch code to be run across any distributed configuration by adding just four lines of code. This makes training and inference at scale simple, efficient and adaptable.

## Installing Accelerate {#installing_accelerate}

Our recommendation is to install it using our provided Python [wheel](https://pythonwheels.com/) as follows:

:   1\. Load a Python [module](https://docs.alliancecan.ca/Utiliser_des_modules/en#Sub-command_load "module"){.wikilink}, thus `module load python`
:   2\. Create and start a [virtual environment](https://docs.alliancecan.ca/Python#Creating_and_using_a_virtual_environment "virtual environment"){.wikilink}.
:   3\. Install Accelerate in the virtual environment with `pip install`.

<!-- -->

:   

## Multi-GPU & multi-node jobs with Accelerate {#multi_gpu_multi_node_jobs_with_accelerate}

In the example that follows, we use `accelerate` to reproduce our [PyTorch tutorial](https://docs.alliancecan.ca/PyTorch#PyTorch_with_Multiple_GPUs "PyTorch tutorial"){.wikilink} on how to train a model with multiple GPUs distributed over multiple nodes. Notable differences are:

:   1\. Here we ask for only one task per node and we let `accelerate` handle starting the appropriate number of processes (one per GPU) on each node.
:   2\. We pass the number of nodes in the job and the individual node ids in the job to accelerate via the `machine_rank` and `num_machines` arguments respectively. Accelerate handles setting global and local ranks internally.

Where the script `config_env.sh` is:

The script `launch_training_accelerate.sh` is:

And finally, `pytorch-accelerate.py` is:

# Training Large Language Models (LLMs) {#training_large_language_models_llms}

The following is a tutorial on how to train LLMs using Huggingface libraries on the Alliance\'s clusters. The goal is to illustrate a set of best practices to make the most out of our infrastructures and avoid common pitfalls. Here we fine-tune [Huggingface's Zephyr model](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta) on the [ultrachat_200k dataset](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k).

## Downloading the Model and the Dataset {#downloading_the_model_and_the_dataset}

The first step is to download the model and the dataset. **On a login node**, we run the following commands to download the model to our project directory using the `git-lfs` module:

`cd projects/account-name/user-name/ `\
`module load git-lfs`\
`git clone `[`https://huggingface.co/HuggingFaceH4/zephyr-7b-beta`](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta)

Next, we create a temporary virtual environment, and we use a python script to download the `ultrachat_200k` dataset to a directory named `ultrachat_dataset` inside our project space:

`module load python/3.11 gcc arrow`\
`virtualenv --no-download ENV`\
`source ENV/bin/activate`\
`pip install --no-index datasets`\
`mkdir ultrachat_dataset `\
`HF_DATASETS_CACHE=ultrachat_dataset python get_ultrachat.py`\
`deactivate`\
`rm -r ENV`

Where the script `get_ultrachat.py` is:

Now that the model and the dataset are both saved locally on the cluster's network filesystem, the next step is to design a job with sufficient resources to train our LLM efficiently. The main factors that might hinder training performance, or prevent the training script from even running in the first place are:

1.  The model is too large to fit entirely inside the memory of a single GPU.
2.  The training set, while relatively small in size, is made up of a large number of very small examples.

To address these factors, our job will be designed to:

1.  Employ a strategy to shard the LLM across multiple GPUs.
2.  Read the dataset from the compute node's local storage as opposed to the cluster's parallel filesystem, and store it in the node's memory afterwards.

To shard the LLM across multiple devices, we will use the `accelerate` library, along with a configuration file describing a [Fully Sharded Data Parallel (FSDP)](https://huggingface.co/docs/transformers/main/en/fsdp#fsdp-configuration) strategy. Using `accelerate`, the sharding strategy is applied automatically, without us having to explicitly write the code to do it inside the training script. To read the dataset from the compute node's local storage, it suffices to copy the dataset over to `$SLURM_TMPDIR`.

For this example, we will reserve a whole node with 4 GPUs and 48 CPUs, such as the GPU nodes available on Narval. The job submission script is then:

Where the config file `fsdp.yaml` is:

And the script `train_llm.py` is:

```{=mediawiki}
{{File
  |name=train_llm.py
  |lang="python"
  |contents=
import transformers
from transformers import MistralForCausalLM, AutoTokenizer
from transformers import TrainingArguments

from datasets import load_dataset
from datasets import DatasetDict

from accelerate import Accelerator

from trl import SFTTrainer

accelerator = Accelerator()

def main():

   dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_gen")

   tokenizer = AutoTokenizer.from_pretrained("./zephyr-7b-beta",local_files_only=True)

   tokenizer.pad_token = tokenizer.eos_token

   model = MistralForCausalLM.from_pretrained("./zephyr-7b-beta",local_files_only=True)

   def preprocess(samples):
      batch = []
      for conversation in samples["messages"]:
         batch.append(tokenizer.apply_chat_template(conversation, tokenize=False))
      return {"content": batch}

   training_set=DatasetDict()

   training_set["train"] = dataset.map(preprocess,
            batched=True,
             remove_columns=dataset.column_names
            )

   trainer = SFTTrainer(
     model=model,
     train_dataset=training_set["train"],
     tokenizer=tokenizer,
     packing=True,
     dataset_text_field="content",
     max_seq_length=2048,
     args=transformers.TrainingArguments(
        output_dir="./",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        max_steps=15,
        logging_steps=1,
        learning_rate=2.5e-5,
        optim="adamw_torch",
        logging_dir="./logs",        # Directory for storing logs
        remove_unused_columns=False,
     ),
   )

   trainer.train()

   accelerator.wait_for_everyone()

if __name__=='__main__':
   main()
}}
```
