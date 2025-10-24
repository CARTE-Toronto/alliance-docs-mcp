---
title: "Large Language Models (LLMs)"
url: "https://docs.alliancecan.ca/wiki/Large_Language_Models_(LLMs)"
category: "General"
last_modified: "2025-09-11T13:02:52Z"
page_id: 31318
display_title: "Large Language Models (LLMs)"
---

Large Language Models (LLMs) are generative models capable of generating sophisticated natural language text and can be used to power conversational systems. On this page, you will find resources and tutorials on how to setup and run LLM training or inference workloads on our systems.

# Downloading Models {#downloading_models}

At the time of this writing, [The Hugging Face Hub](https://huggingface.co/models) is the most common repository for LLMs.

The `huggingface_hub` Python package contains a command line interface (CLI) which can be used to download models. For example, to download the model `Zephyr-7b-beta`, first install `huggingface_hub` in a [virtual environment](https://docs.alliancecan.ca/Virtualenv "virtual environment"){.wikilink}, then **on a login node** run:

` HF_HUB_DISABLE_XET=1 hf download --max-workers=1 HuggingFaceH4/zephyr-7b-beta`

Note that we set the variable `HF_HUB_DISABLE_XET` to avoid using the `hf_xet` package to download models. This package, meant to make downloading artifacts from the Hugging Face more efficient, currently leads to failures on our systems and should not be used at this time.

For more options, please see our article on the [Hugging Face](https://docs.alliancecan.ca/Huggingface "Hugging Face"){.wikilink} ecosystem.

# Inference

# Training
