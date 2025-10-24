---
title: "VLLM"
url: "https://docs.alliancecan.ca/wiki/VLLM"
category: "General"
last_modified: "2025-05-29T13:54:55Z"
page_id: 28589
display_title: "VLLM"
---

[vLLM](https://github.com/vllm-project/vllm) is a community-driven project that provides high-throughput and memory-efficient inference and serving for large language models (LLMs). It supports various decoding algorithms, quantizations, parallelism, and models from Hugging Face and other sources.

# Installation

## Latest available wheels {#latest_available_wheels}

To see the latest version of vLLM that we have built:

For more information, see [Available wheels](https://docs.alliancecan.ca/Python#Available_wheels "Available wheels"){.wikilink}.

## Installing our wheel {#installing_our_wheel}

The preferred option is to install it using the Python [wheel](https://pythonwheels.com/) as follows:

:   1\. Load dependencies, load a Python and OpenCV [modules](https://docs.alliancecan.ca/Utiliser_des_modules/en#Sub-command_load "modules"){.wikilink},

<!-- -->

:   2\. Create and start a temporary [virtual environment](https://docs.alliancecan.ca/Python#Creating_and_using_a_virtual_environment "virtual environment"){.wikilink}.

<!-- -->

:   3\. Install vLLM in the virtual environment and its Python dependencies.

X.Y.Z }} where `X.Y.Z` is the exact desired version, for instance `0.8.4`. You can omit to specify the version in order to install the latest one available from the wheelhouse.

:   4\. Freeze the environment and requirements set.

<!-- -->

:   5\. Deactivate the environment.

<!-- -->

:   6\. Clean up and remove the virtual environment.

# Job submission {#job_submission}

## Before submitting a job: Downloading models {#before_submitting_a_job_downloading_models}

Models loaded for inference on vLLM will typically come from the [Hugging Face Hub](https://huggingface.co/docs/hub/models-the-hub).

The following is an example of how to use the command line tool from the Hugging face to download a model. Note that models must be downloaded on a login node to avoid idle compute while waiting for resources to download. Also note that models will be cached at by default at `$HOME/.cache/huggingface/hub`. For more information on how to change the default cache location, as well as other means of downloading models, please see our article on the [Hugging Face ecosystem](https://docs.alliancecan.ca/Huggingface "Hugging Face ecosystem"){.wikilink}.

`module load python/3.12`\
`virtualenv --no-download temp_env && source temp_env/bin/activate`\
`pip install --no-index huggingface_hub`\
`huggingface-cli download facebook/opt-125m`\
`rm -r temp_env`

## Single Node {#single_node}

The following is an example of how to submit a job that performs inference on a model split across 2 GPUs. If your model **fits entirely inside one GPU**, change the python script below to call `LLM(``<model name>`{=html}`)` without extra arguments.

This example **assumes you have pre-downloaded** the model `facebook/opt-125m` as described on the previous section.

```{=mediawiki}
{{File
  |name=vllm-example.py
  |lang="python"
  |contents=

from vllm import LLM

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

# Set "tensor_parallel_size" to the number of GPUs in your job.

llm = LLM(model="facebook/opt-125m",tensor_parallel_size=2)

outputs = llm.generate(prompts)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
}}
```
## Multiple Nodes {#multiple_nodes}

The following example revisits the single node example above, but splits the model across 4 GPUs over 2 separate nodes, i.e., 2 GPUs per node.

Currently, vLLM relies on [Ray](https://docs.alliancecan.ca/Ray "Ray"){.wikilink} to manage splitting models over multiple nodes. The code example below contains the necessary steps to start a [multi-node Ray cluster](https://docs.alliancecan.ca/Ray#Multiple_Nodes "multi-node Ray cluster"){.wikilink} and run vLLM on top of it:

Where the script `config_env.sh` is:

The script `launch_ray.sh` is:

```{=mediawiki}
{{File
  |name=launch_ray.sh
  |lang="bash"
  |contents=
#!/bin/bash

if [[ "$SLURM_PROCID" -eq "0" ]]; then
        echo "Ray head node already started..."
        sleep 10

else
        export VLLM_HOST_IP=`hostname --ip-address`
        ray start --address "${HEAD_NODE}:${RAY_PORT}" --num-cpus="${SLURM_CPUS_PER_TASK}" --num-gpus=2 --block
        sleep 5
        echo "ray worker started!"
fi
}}
```
And finally, the script `vllm_example.py` is:

```{=mediawiki}
{{File
  |name=vllm_example.py
  |lang="python"
  |contents=

from vllm import LLM

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

# Set "tensor_parallel_size" to the TOTAL number of GPUs on all nodes.

llm = LLM(model="facebook/opt-125m",tensor_parallel_size=4)

outputs = llm.generate(prompts)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
}}
```
