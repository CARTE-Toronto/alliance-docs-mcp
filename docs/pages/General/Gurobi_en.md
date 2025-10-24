---
title: "Gurobi/en"
url: "https://docs.alliancecan.ca/wiki/Gurobi/en"
category: "General"
last_modified: "2025-01-20T19:16:02Z"
page_id: 14002
display_title: "Gurobi"
---

`<languages />`{=html}

[Gurobi](http://www.gurobi.com/) is a commercial software suite for solving complex optimization problems. This wiki page describes the non-commercial use of Gurobi software on our clusters.

## License limitations {#license_limitations}

We support and provide a free license to use Gurobi on the [Graham](https://docs.alliancecan.ca/Graham "Graham"){.wikilink}, [Cedar](https://docs.alliancecan.ca/Cedar "Cedar"){.wikilink}, [Béluga](https://docs.alliancecan.ca/Beluga/en "Béluga"){.wikilink} and [Niagara](https://docs.alliancecan.ca/Niagara "Niagara"){.wikilink} clusters. The license provides a total number of 4096 simultaneous uses (tokens in use) and permits distributed optimization with up to 100 nodes. A single user can run multiple simultaneous jobs. In order to use Gurobi, you must agree to certain conditions. Please [ contact support](https://docs.alliancecan.ca/Technical_support " contact support"){.wikilink} and include a copy of the following completed agreement. You will then be added into our license file as a user within a few days.

### Academic usage agreement {#academic_usage_agreement}

My Alliance username is \"\_\_\_\_\_\_\_\" and I am a member of the academic institution \"\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_.\" This message confirms that I will only use the Gurobi license provided on Digital Research Alliance of Canada systems for the purpose of non-commercial research project(s) to be published in publicly available article(s).

### Configuring your account {#configuring_your_account}

You do NOT need to create a `~/.licenses/gurobi.lic` file. The required settings to use our Gurobi license are configured by default when you load a Gurobi module on any cluster. To verify your username has been added to our Gurobi license and is working properly, run the following command:

`$ module load gurobi`\
`$ gurobi_cl 1> /dev/null && echo Success || echo Fail`

If it returns \"Success\" you can begin using Gurobi immediately. If the test returns \"Fail\" then check whether a file named `<i>`{=html}\~/.license/gurobi`</i>`{=html} exists; if so, rename or remove this file, reload the module and try the test again. If it still returns \"Fail\", check whether there are any environment variables containing GUROBI defined in either of our your `<i>`{=html}\~/.bashrc`</i>`{=html} or `<i>`{=html}\~/.bash_profile`</I>`{=html} files. If you find any, comment or remove those lines, log out and log in again, reload the Gurobi module and try the test again. If you still get \"Fail\", [ contact support](https://docs.alliancecan.ca/Technical_support " contact support"){.wikilink} for help.

### Minimizing license checkouts {#minimizing_license_checkouts}

Note that all Gurobi license checkouts are handled by a single license server located in Ontario; it is therefore important to limit license checkout attempts as much as possible. Rather than checking out a license for each invocation of Gurobi in a job\-\--which may occur dozens or even hundreds of times\-\--you should ensure that your program, whatever the language or computing environment used, only makes a single license checkout and then reuses this license token throughout the lifetime of the job. This will improve your job\'s performance because contacting a remote license server is very costly in time; moreover, responsiveness of our license server for everyone using Gurobi will also improve. `<span style="color:red">`{=html}Failure to use Gurobi carefully in this regard may ultimately result in random intermittent license checkout failures for all users. If this happens, you will be contacted and asked to kill all your jobs until your program is fixed and tested to ensure the problem is resolved.`</span>`{=html} Some documentation on this subject for C++ programs may be found [here](https://www.gurobi.com/documentation/9.5/refman/cpp_env2.html), explaining how to create a single Gurobi environment which can then be used for all your models. Python users can consult this [page](https://www.gurobi.com/documentation/9.5/refman/py_env_start.html), which discusses how to implement this same idea of using a single environment and thus a single license token with multiple models. Other programs that call Gurobi, such as R, can also easily trigger the problem when run in parallel, especially when many simultaneous parallel jobs are submitted and/or run.

## Interactive allocations {#interactive_allocations}

### Gurobi command-line tools {#gurobi_command_line_tools}

`[gra-login2:~] salloc --time=1:00:0 --cpus-per-task=8 --mem=1G --account=def-xyz`\
`[gra800:~] module load gurobi`\
`[gra800:~] gurobi_cl Record=1 Threads=8 Method=2 ResultFile=p0033.sol LogFile=p0033.log $GUROBI_HOME/examples/data/p0033.mps`\
`[gra800:~] gurobi_cl --help`

### Gurobi interactive shell {#gurobi_interactive_shell}

`[gra-login2:~] salloc --time=1:00:0 --cpus-per-task=8 --mem=1G --account=def-xyz`\
`[gra800:~] module load gurobi`\
`[gra800:~] echo "Record 1" > gurobi.env    see *`\
`[gra800:~] gurobi.sh`\
`gurobi> m = read('/cvmfs/restricted.computecanada.ca/easybuild/software/2017/Core/gurobi/8.1.1/examples/data/glass4.mps') `\
`gurobi> m.Params.Threads = 8               see **`\
`gurobi> m.Params.Method = 2`\
`gurobi> m.Params.ResultFile = "glass4.sol"`\
`gurobi> m.Params.LogFile = "glass4.log"`\
`gurobi> m.optimize()`\
`gurobi> m.write('glass4.lp')`\
`gurobi> m.status                           see ***`\
`gurobi> m.runtime                          see ****`\
`gurobi> help()`

where

`   * `[`https://www.gurobi.com/documentation/8.1/refman/recording_api_calls.html`](https://www.gurobi.com/documentation/8.1/refman/recording_api_calls.html)\
`  ** `[`https://www.gurobi.com/documentation/8.1/refman/parameter_descriptions.html`](https://www.gurobi.com/documentation/8.1/refman/parameter_descriptions.html)\
` *** `[`https://www.gurobi.com/documentation/8.1/refman/optimization_status_codes.html`](https://www.gurobi.com/documentation/8.1/refman/optimization_status_codes.html)\
`**** `[`https://www.gurobi.com/documentation/8.1/refman/attributes.html`](https://www.gurobi.com/documentation/8.1/refman/attributes.html)

### Replaying API calls {#replaying_api_calls}

You can record API calls and repeat them with

`[gra800:~] gurobi_cl recording000.grbr`

Reference: <https://www.gurobi.com/documentation/8.1/refman/recording_api_calls.html>

## Cluster batch job submission {#cluster_batch_job_submission}

Once a Slurm script has been prepared for a Gurobi problem, it can be submitted to the queue by running the `sbatch script-name.sh` command. The jobs status in the queue can then be checked by running the `sq` command. The following Slurm scripts demonstrate solving 2 problems provided in the `examples` directory of each Gurobi module.

### Data example {#data_example}

The following Slurm script utilizes the [Gurobi command-line interface](https://www.gurobi.com/documentation/9.5/quickstart_linux/solving_the_model_using_th.html) to solve a [simple coin production model](https://www.gurobi.com/documentation/9.5/quickstart_linux/solving_a_simple_model_the.html) written in [LP format](https://www.gurobi.com/documentation/9.5/refman/lp_format.html). The last line demonstrates how [parameters](https://www.gurobi.com/documentation/9.5/refman/parameters.html) can be passed directly to the Gurobi command-line tool `gurobi_cl` using simple command line arguments. For help selecting which [parameters](https://www.gurobi.com/documentation/9.5/refman/parameters.html) are best used for a particular problem and for choosing optimal values, refer to both the `<i>`{=html}Performance and Parameters`</i>`{=html} and `<i>`{=html}Algorithms and Search`</I>`{=html} sections found in the [Gurobi Knowledge Base](https://support.gurobi.com/hc/en-us/categories/360000840331-Knowledge-Base) as well as the extensive online [Gurobi documentation](https://www.gurobi.com/documentation/). `{{File
  |name=script-lp_coins.sh
  |lang="sh"
  |contents=
#!/bin/bash
#SBATCH --account=def-group   # some account
#SBATCH --time=0-00:30        # specify time limit (D-HH:MM)
#SBATCH --cpus-per-task=8     # specify number threads
#SBATCH --mem=4G              # specify total memory
#SBATCH --nodes=1             # do not change

#module load StdEnv/2016      # for versions < 9.0.3 
module load StdEnv/2020       # for versions > 9.0.2

module load gurobi/9.5.0

rm -f coins.sol
gurobi_cl Threads=$SLURM_CPUS_ON_NODE Method=2 ResultFile=coins.sol ${GUROBI_HOME}/examples/data/coins.lp
}}`{=mediawiki}

### Python example {#python_example}

This is an example Slurm script for solving a [simple facility location model](https://www.gurobi.com/documentation/9.5/examples/a_list_of_the_grb_examples.html) with [Gurobi Python](https://www.gurobi.com/documentation/9.5/examples/facility_py.html). The example shows how to set the threads [parameter](https://www.gurobi.com/documentation/9.5/refman/parameters.html#sec:Parameters) equal to the number of cores allocated to a job by dynamically generating a [gurobi.env](https://www.gurobi.com/documentation/9.5/quickstart_linux/using_a_grb_env_file.html) file into the working directory when using the [Gurobi python interface](https://www.gurobi.com/documentation/9.5/refman/python_parameter_examples.html). This must be done for each submitted job, otherwise Gurobi will (by default) start as many execute [threads](https://www.gurobi.com/documentation/9.5/refman/threads.html#parameter:Threads) as there are physical cores on the compute node potentially slowing down the job and negatively impacting other user jobs running on the same node. `{{File
  |name=script-facility.sh
  |lang="sh"
  |contents=
#!/bin/bash
#SBATCH --account=def-group   # some account
#SBATCH --time=0-00:30        # specify time limit (D-HH:MM)
#SBATCH --cpus-per-task=4     # specify number threads
#SBATCH --mem=4G              # specify total memory
#SBATCH --nodes=1             # do not change

#module load StdEnv/2020      # for versions < 10.0.3
module load StdEnv/2023       # for versions > 10.0.3

module load gurobi/11.0.1

echo "Threads ${SLURM_CPUS_ON_NODE:-1}" > gurobi.env

gurobi.sh ${GUROBI_HOME}/examples/python/facility.py
}}`{=mediawiki}

## Using Gurobi in Python virtual environments {#using_gurobi_in_python_virtual_environments}

Gurobi brings it\'s own version of Python which does not contain any 3rd-party Python packages except Gurobi. In order to use Gurobi together with popular Python packages like NumPy, Matplotlib, Pandas and others, we need to create a [virtual Python environment](https://docs.alliancecan.ca/Python#Creating_and_using_a_virtual_environment "virtual Python environment"){.wikilink} in which we can install both `gurobipy` and for example `pandas`. Before we start, we need to decide which combination of versions for Gurobi and Python to use. Following is a list of the Python versions supported by the major Gurobi versions installed in the previous through current standard environments (StdEnv):

`[name@server ~] module load StdEnv/2016; module load gurobi/8.1.1; cd $EBROOTGUROBI/lib; ls -d python*`\
`python2.7  python2.7_utf16  python2.7_utf32  python3.5_utf32  python3.6_utf32  python3.7_utf32`\
` `\
`[name@server ~] module load StdEnv/2020; module load gurobi/9.5.2; cd $EBROOTGUROBI/lib; ls -d python*`\
`python2.7_utf16  python2.7_utf32  python3.10_utf32  python3.7  python3.7_utf32  python3.8_utf32  python3.9_utf32`\
\
`[name@server ~] module load StdEnv/2023; module load gurobi/10.0.3; cd $EBROOTGUROBI/lib; ls -d python*`\
`python3.10_utf32  python3.11_utf32  python3.7  python3.7_utf32  python3.8_utf32  python3.9_utf32`\
\
`[name@server ~] module load StdEnv/2023; module load gurobi/11.0.1; cd $EBROOTGUROBI/lib; ls -d python*`\
`python3.11`

### Installing Gurobi for Python {#installing_gurobi_for_python}

As mentioned near the end of this official document [How do I install Gurobi for Python?](http://support.gurobi.com/hc/en-us/articles/360044290292-How-do-I-install-Gurobi-for-Python), the previously recommended method for installing Gurobi for Python with `setup.py` has been deprecated to only be usable with Gurobi 10 versions (and older). Section `<i>`{=html}Gurobi 11 versions (and newer)`</i>`{=html} below shows how to simultaneously download a compatible binary wheel from [pypi.org](https://pypi.org/project/gurobipy/) and convert it into a format usable with the newly recommended command to install Gurobi for Python.

### Gurobi versions 10.0.3 (and older) {#gurobi_versions_10.0.3_and_older}

The following steps need to be done once per system and are usable with StdEnv/2023 and older. First, load the modules to [create the virtual environment](https://docs.alliancecan.ca/Python#Creating_and_using_a_virtual_environment "create the virtual environment"){.wikilink} and activate it:

Now install any Python packages you want to use, in this case `pandas`:

Next, install gurobipy in the environment. Note that as of StdEnv/2023 the installation can no longer be done under \$EBROOTGUROBI using the command `python setup.py build --build-base /tmp/${USER} install` since a fatal error (`error: could not create 'gurobipy.egg-info': Read-only file system`) will occur. Instead, the required files need to be copied elsewhere (such as /tmp/\$USER) and the installation made from there, for example:

```{=mediawiki}
{{Commands|prompt=(env_gurobi) [name@server ~] $
| mkdir /tmp/$USER
| cp -r $EBROOTGUROBI/{lib,setup.py} /tmp/$USER
| cd /tmp/$USER
| python setup.py install
(env_gurobi) [roberpj@gra-login1:/tmp/roberpj] python setup.py install
/home/roberpj/env_gurobi/lib/python3.11/site-packages/setuptools/_core_metadata.py:158: SetuptoolsDeprecationWarning: Invalid config.
!!

        ********************************************************************************
        newlines are not allowed in `summary` and will break in the future
        ********************************************************************************

!!
  write_field('Summary', single_line(summary))
removing /tmp/roberpj/build
(env_gurobi) [roberpj@gra-login1:/tmp/roberpj]
| deactivate
[name@server ~]
}}
```
### Gurobi versions 11.0.0 (and newer) {#gurobi_versions_11.0.0_and_newer}

Once again, the following steps need to be done once per system and are usable with StdEnv/2023 and older. First load the modules to [create the virtual environment](https://docs.alliancecan.ca/Python#Creating_and_using_a_virtual_environment "create the virtual environment"){.wikilink} and activate it. Version 11.0.0 is skipped since it has been observed to seg fault in at least one example versus Version 11.0.1 which runs smoothly.

As before, install any needed Python packages. Since the following matrix example requires `numpy`, we install the pandas package:

Next install gurobipy into the environment. As mentioned above and in [this article](https://support.gurobi.com/hc/en-us/articles/360044290292-How-do-I-install-Gurobi-for-Python "this article"){.wikilink} the use of setup.py to install Gurobi for python is deprecated starting with Gurobi 11. Both pip and conda are given as alternatives; however, since conda should not be used on our systems, the pip approach will be demonstrated here. The installation of gurobipy is slightly complicated since our Linux systems are set up with gentoo prefix. As a result neither A) the recommended command to download and install the gurobipy extension from the public PyPI server `pip install gurobipy==11.0.1` mentioned in the article line or B) the offline command to install the wheel with `python -m pip install --find-links ``<wheel-dir>`{=html}` --no-index gurobipy`, will work. Instead, we have prepared a script to download and simultaneously convert the existing wheel into a usable format with a new name. There is one caveat; for each new Gurobi version, you must go into <https://pypi.org/project/gurobipy/11.0.1/#history> and click on the desired version followed by the `Download files` button located in the menu on the left. Finally, click to copy the https link for the wheel file (named gurobipy-11.0.1-cp311-cp311-manylinux2014_x86_64.manylinux_2_17_x86_64.whl in the case of Gurobi 11.0.1) and paste it as the \--url argument as shown below :

### Running Gurobi in the environment {#running_gurobi_in_the_environment}

Once created our Gurobi environment can be activated and used at any time. To demonstrate this we also load gurobi (so \$EBROOTGUROBI is defined) and `scipy-stack` (so scipy is available). Both are required to run the matrix example (along with numpy that was already installed into our environment with pip in a previous step above via pandas).

Python scripts, such as the examples provided with the gurobi module can now be run (within the virtual environment) using python :

Likewise custom python scripts such as the following can be run as jobs in the queue by writing slurm scripts that load your virtual environment.

Submit your script to the queue by running `sbatch my_slurm_script.sh` as per usual :

```{=mediawiki}
{{File
  |name=my_slurm_script.sh
  |lang="sh"
  |contents=
#!/bin/bash
#SBATCH --account=def-somegrp  # specify an account
#SBATCH --time=0-00:30         # time limit (D-HH:MM)
#SBATCH --nodes=1              # run job on one node
#SBATCH --cpus-per-task=4      # specify number of CPUS
#SBATCH --mem=4000M            # specify total MB memory

module load StdEnv/2023
module load gurobi/11.0.1
# module load scipy-stack      # uncomment if needed

echo "Threads ${SLURM_CPUS_ON_NODE:-1}" > gurobi.env

source ~/env_gurobi/bin/activate
python my_gurobi_script.py
}}
```
Further information regarding how to create and use python virtual environments within job scripts can be found [here](https://docs.alliancecan.ca/Python#Creating_virtual_environments_inside_of_your_jobs "here"){.wikilink}.

## Using Gurobi with Java {#using_gurobi_with_java}

To use Gurobi with Java, you will also need to load a Java module and add an option to your Java command in order to allow the Java virtual environment to find the Gurobi libraries. A sample job script is below:

## Using Gurobi with Jupyter notebooks {#using_gurobi_with_jupyter_notebooks}

Various topics can be found by visiting [Resources](https://www.gurobi.com/resources/), then clicking [Code and Modeling Examples](https://www.gurobi.com/resources/?category-filter=code-example) and finally [Optimization with Python -- Jupyter Notebook Modeling Examples](https://www.gurobi.com/resource/modeling-examples-using-the-gurobi-python-api-in-jupyter-notebook/). Alternatively visit [support.gurobi.com](https://support.gurobi.com/) and search on `<I>`{=html}Jupyter Notebooks`</I>`{=html}.

A demo case of using Gurobi with Jupyter notebooks on our systems can be found in this [video recording, i.e. at time 38:28](https://youtu.be/Qk3Le5HBxeg?t=2310).

## Cite Gurobi {#cite_gurobi}

Please see [How do I cite Gurobi software for an academic publication?](https://support.gurobi.com/hc/en-us/articles/360013195592-How-do-I-cite-Gurobi-software-for-an-academic-publication-)
