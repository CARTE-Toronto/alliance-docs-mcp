---
title: "Interpretable AI"
url: "https://docs.alliancecan.ca/wiki/Interpretable_AI"
category: "General"
last_modified: "2025-12-06T14:07:27Z"
page_id: 32081
display_title: "Interpretable AI"
---

[Interpretable AI](http://interpretable.ai) (IAI) is a suite of machine learning tools that aim for full transparency and understand-ability by humans. IAI provides a python interface for their algorithms and can only be used under license. IAI provides a free license for academic use which can be accessed on Alliance systems. For licensing issues please contact [technical support](https://docs.alliancecan.ca/wiki/Technical_support).

## Installation on Alliance clusters {#installation_on_alliance_clusters}

1\. Go to the [IAI download page](https://docs.interpretable.ai/stable/download/) and download the latest release of IAI. Be sure to select \"Linux\" and the correct version of Julia (1.11.3 in these instructions).

2\. Upload the files from Step 1 to the cluster of your choice. Unzip the file, note the location of the files. There should be a `sys.so` and `Artifacts.toml` file in the directory. In this example we\'ll say `/path/to/sysimage/`

3\. Load requisite modules:

`module load StdEnv/2023 python/3.11.5 scipy-stack/2025a julia/1.11.3`

4\. Make a python virtual environment and install IAI:

`virtualenv --no-download iaiENV && source iaiENV/bin/activate`\
`pip install --no-index --upgrade pip`\
`pip install interpretableai`

5\. Install IAI in julia and build PyCall to work in your virtual environment. Make sure your virtual environment is active before launching Julia:

`(iaiENV) $ julia`\
`[...]`\
`julia`\
`julia> import Pkg; Pkg.add(url="`[`https://github.com/InterpretableAI/IAISystemImages.jl`](https://github.com/InterpretableAI/IAISystemImages.jl)`")`\
`julia> import IAISystemImages`\
`julia> IAISystemImages.install_artifacts("/path/to/sysimage/Artifacts.toml")`\
`julia> using Pkg, PyCall`\
`julia> ENV["PYTHON"] = joinpath(ENV["VIRTUAL_ENV"], "bin", "python")`\
`julia> Pkg.build("PyCall")`\
`julia> exit()`

6\. Finish IAI setup in Python:

`(iaiENV) $ python`\
`[...]`\
`from julia import Julia`\
`Julia(sysimage='/path/to/sysimage/sys.so')`\
`import interpretableai`\
`interpretableai.install()`

7\. Import IAI for use in your python scripts

`from interpretableai import iai`

### IMPORTANT NOTE {#important_note}

Any Python script using the `interpretableai` package has to include setting up Julia to run with the IAI system image. This means the following code must be included before importing IAI in your Python scripts.

`from julia import Julia`\
`Julia(sysimage='/path/to/sysimage/sys.so')`

It is also a good idea to disable automatic updating in your job script:

`export IAI_DISABLE_UPDATE_CHECK=true`
