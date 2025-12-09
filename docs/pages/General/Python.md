---
title: "Python/en"
url: "https://docs.alliancecan.ca/wiki/Python/en"
category: "General"
last_modified: "2025-09-18T12:26:01Z"
page_id: 2586
display_title: "Python"
---

`<languages />`{=html}

## Description

[Python](http://www.python.org/) is an interpreted programming language with a design philosophy stressing the readability of code. Its syntax is simple and expressive. Python has an extensive, easy-to-use standard library.

The capabilities of Python can be extended with packages developed by third parties. In general, to simplify operations, it is left up to individual users and groups to install these third-party packages in their own directories. However, most systems offer several versions of Python as well as tools to help you install the third-party packages that you need.

The following sections discuss the Python interpreter, and how to install and use packages.

## Loading an interpreter {#loading_an_interpreter}

### Default Python version {#default_python_version}

When you log into our clusters, a default Python version will be available, but that is generally not the one that you should use, especially if you need to install any Python packages. You should try to find out which version of Python is required to run your Python programs and load the appropriate [ module](https://docs.alliancecan.ca/Utiliser_des_modules/en " module"){.wikilink}. If you are not sure which version you need, then it is reasonable to use the latest version available.

### Loading a Python module {#loading_a_python_module}

To discover the versions of Python available:

You can then load the version of your choice using `module load`. For example

where `X.Y` represent the version, for example `3.13`.

### Python version supported {#python_version_supported}

In general in the Python ecosystem, the transition to more modern versions of python is accelerating, with many packages only supporting the latest few versions of Python 3.x. In our case, we provide prebuilt Python packages in our [wheelhouse](https://docs.alliancecan.ca/Available_Python_wheels "wheelhouse"){.wikilink} only for the 3 most recent Python versions available on the systems. This will result in dependencies issues when trying to install those packages with older versions of Python. See [Troubleshooting](https://docs.alliancecan.ca/Python#Package_.27X.27_requires_a_different_Python:_X.Y.Z_not_in_.27.3E.3DX.Y.27 "Troubleshooting"){.wikilink}.

### SciPy stack {#scipy_stack}

In addition to the base Python module, the [SciPy](https://www.scipy.org/) package is also available as an [environment module](https://docs.alliancecan.ca/Utiliser_des_modules/en "environment module"){.wikilink}. The `scipy-stack` module includes:

- NumPy
- SciPy
- Matplotlib
  - dateutil
  - pytz
- IPython
  - pyzmq
  - tornado
- pandas
- Sympy
- nose

If you want to use any of these Python packages, load a Python version of your choice and then `module load scipy-stack`.

To get a complete list of the packages contained in `scipy-stack`, along with their version numbers, run `module spider scipy-stack/2020a` (replacing `2020a` with whichever version you want to find out about).

## Creating and using a virtual environment {#creating_and_using_a_virtual_environment}

With each version of Python, we provide the tool [virtualenv](http://pypi.python.org/pypi/virtualenv). This tool allows users to create virtual environments within which you can easily install Python packages. These environments allow one to install many versions of the same package, for example, or to compartmentalize a Python installation according to the needs of a specific project. Usually you should create your Python virtual environment(s) in your /home directory or in one of your /project directories. (See \"Creating virtual environments inside of your jobs\" below for a third alternative.)

To create a virtual environment, make sure you have selected a Python version with `module load python/X.Y.Z` as shown above in section *Loading a Python module*. If you expect to use any of the packages listed in section *SciPy stack* above, also run `module load scipy-stack/X.Y.Z`. Then enter the following command, where `ENV` is the name of the directory for your new environment:

Once the virtual environment has been created, it must be activated:

You should also upgrade `pip` in the environment:

To exit the virtual environment, simply enter the command `deactivate`:

You can now use the same virtual environment over and over again. Each time:

1.  Load the same environment modules that you loaded when you created the virtual environment, e.g. `module load python scipy-stack`
2.  Activate the environment, `source ENV/bin/activate`

### Installing packages {#installing_packages}

Once you have a virtual environment loaded, you will be able to run the [pip](http://www.pip-installer.org/) command. This command takes care of compiling and installing most of Python packages and their dependencies. A comprehensive index of Python packages can be found at [PyPI](https://pypi.python.org/pypi).

All of `pip`\'s commands are explained in detail in the [user guide](https://pip.pypa.io/en/stable/user_guide/). We will cover only the most important commands and use the [Numpy](http://numpy.scipy.org/) package as an example.

We first load the Python interpreter:

where `X.Y` represent the version, for example `3.13`.

We then activate the virtual environment, previously created using the `virtualenv` command:

Finally, we install the latest stable version of Numpy:

The `pip` command can install packages from a variety of sources, including PyPI and prebuilt distribution packages called Python [wheels](https://pythonwheels.com/). We provide Python wheels for a number of packages. In the above example, the [`--no-index`](https://pip.pypa.io/en/stable/reference/pip_wheel/#cmdoption-no-index) option tells `pip` to *not* install from PyPI, but instead to install only from locally available packages, i.e. our wheels.

Whenever we provide a wheel for a given package, we strongly recommend to use it by way of the `--no-index` option. Compared to using packages from PyPI, wheels that have been compiled by our staff can prevent issues with missing or conflicting dependencies, and were optimized for our clusters hardware and libraries. See [Available wheels](https://docs.alliancecan.ca/#Available_wheels "Available wheels"){.wikilink}.

If you omit the `--no-index` option, `pip` will search both PyPI and local packages, and use the latest version available. If PyPI has a newer version, it will be installed instead of our wheel, possibly causing issues. If you are certain that you prefer to download a package from PyPI rather than use a wheel, you can use the `--no-binary` option, which tells `pip` to ignore prebuilt packages entirely. Note that this will also ignore wheels that are distributed through PyPI, and will always compile the package from source.

To see where the `pip` command is installing a python package from, diagnosing installation issues, you can tell it to be more verbose with the `-vvv` option. It is also worth mentioning that when installing multiple packages it is advisable to install them with one command as it helps pip resolve dependencies.

### Creating virtual environments inside of your jobs {#creating_virtual_environments_inside_of_your_jobs}

**Note**: On Trillium it is recommended to create virtual environments from a login node in `HOME` and source it in your job script.

Parallel filesystems such as the ones used on our clusters are very good at reading or writing large chunks of data, but can be bad for intensive use of small files. Launching a software and loading libraries, such as starting Python and loading a virtual environment, can be slow for this reason.

As a workaround for this kind of slowdown, and especially for single-node Python jobs, you can create your virtual environment inside of your job, using the compute node\'s local disk. It may seem counter-intuitive to recreate your environment for every job, but it can be faster than running from the parallel filesystem, and will give you some protection against some filesystem performance issues. This approach, of creating a node-local virtualenv, has to be done for each node in the job, since the virtualenv is only accessible on one node. Following job submission script demonstrates how to do this for a single-node job:

where the `requirements.txt` file will have been created from a test environment. For example, if you want to create an environment for [TensorFlow](https://docs.alliancecan.ca/TensorFlow "TensorFlow"){.wikilink}, you would do the following on a login node : /tmp/\$RANDOM \|virtualenv \--no-download \$ENVDIR \|source \$ENVDIR/bin/activate \|pip install \--no-index \--upgrade pip \|pip install \--no-index tensorflow \|pip freeze \--local \> requirements.txt \|deactivate \|rm -rf \$ENVDIR }}

This will yield a file called `requirements.txt`, with content such as the following

This file will ensure that your environment is reproducible between jobs.

Note that the above instructions require all of the packages you need to be available in the python wheels that we provide (see \"Available wheels\" below). If the wheel is not available in our wheelhouse, you can pre-download it (see \"Pre-downloading packages\" section below). If you think that the missing wheel should be included in our wheelhouse, please contact [Technical support](https://docs.alliancecan.ca/Technical_support "Technical support"){.wikilink} to make a request.

#### Creating virtual environments inside of your jobs (multi-nodes) {#creating_virtual_environments_inside_of_your_jobs_multi_nodes}

In order to run scripts across multiple nodes, each node must have its own virtual environment activated.

1\. In your submission script, create the virtual environment on each allocated node:

``` bash
srun --ntasks $SLURM_NNODES --tasks-per-node=1 bash << EOF

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

pip install --no-index --upgrade pip
pip install --no-index -r requirements.txt

EOF
```

2\. Activate the virtual environment on the main node,

``` bash
source $SLURM_TMPDIR/env/bin/activate;
```

3\. Use `srun` to run your script

``` bash
srun python myscript.py;
```

#### Example (multi-nodes) {#example_multi_nodes}

### Available wheels {#available_wheels}

Currently available wheels are listed on the [Available Python wheels](https://docs.alliancecan.ca/Available_Python_wheels "Available Python wheels"){.wikilink} page. You can also run the command `avail_wheels` on the cluster. By default, it will:

- only show you the `<b>`{=html}latest version`</b>`{=html} of a specific package (unless versions are given);
- only show you versions that are compatible with the python module (if one loaded) or virtual environment (if activated), otherwise all versions will be shown;
- only show you versions that are compatible with the CPU architecture and software environment (StdEnv) that you are currently running on.

#### Names

To list wheels containing `cdf` (case insensitive) in its name:

Or an exact name:

#### Version

To list a specific version, you can use the same format as with \`pip\`: 1.23 \|result= name version python arch \-\-\-\-\-- \-\-\-\-\-\-\-\-- \-\-\-\-\-\-\-- \-\-\-\-\-\-- numpy 1.23.0 cp39 generic numpy 1.23.0 cp38 generic numpy 1.23.0 cp310 generic }} Or use the long option:

With the `pip` format, you can use different operators : `==`, `<`, `>`, `~=`, `<=`,`>=`, `!=`. For instance, to list inferior versions:

And to list all available versions:

#### Python

You can list a specific version of Python:

The `<i>`{=html}python`</i>`{=html} column tells us for which version the wheel is available, where `cp39` stands for `cpython 3.9`.

#### Requirements file {#requirements_file}

One can list available wheels based on a `requirements.txt` file with:

And display wheels that are not available:

### Pre-downloading packages {#pre_downloading_packages}

Here is how to pre-download a package called `tensorboardX` on a login node, and install it on a compute node:

1.  Run `pip download --no-deps tensorboardX`. This will download the package as `tensorboardX-1.9-py2.py3-none-any.whl` (or similar) in the working directory. The syntax of `pip download` is the same as `pip install`.
2.  If the filename does not end with `none-any`, and ends with something like `linux_x86_64` or `manylinux*_x86_64`, the wheel might not function correctly. You should contact [Technical support](https://docs.alliancecan.ca/Technical_support "Technical support"){.wikilink} so that we compile the wheel and make it available on our systems.
3.  Then, when installing, use the path for file `pip install tensorboardX-1.9-py2.py3-none-any.whl`.

## Parallel programming with the Python `multiprocessing` module {#parallel_programming_with_the_python_multiprocessing_module}

Doing parallel programming with Python can be an easy way to get results faster. A usual way of doing so is to use the [`multiprocessing`](https://sebastianraschka.com/Articles/2014_multiprocessing.html) module. Of particular interest is the `Pool` class of this module, since it allows one to control the number of processes started in parallel, and apply the same calculation to multiple data. As an example, suppose we want to calculate the `cube` of a list of numbers. The serial code would look like this : `<tabs>`{=html} `<tab name="Using a loop">`{=html}

`</tab>`{=html} `<tab name="Using a map">`{=html}

`</tab>`{=html} `</tabs>`{=html}

Using the `Pool` class, running in parallel, the above codes become : `<tabs>`{=html} `<tab name="Using a loop">`{=html}

`</tab>`{=html} `<tab name="Using a map">`{=html}

`</tab>`{=html} `</tabs>`{=html}

The above examples will however be limited to using `4` processes. On a cluster, it is very important to use the cores that are allocated to your job. Launching more processes than you have cores requested will slow down your calculation and possibly overload the compute node. Launching fewer processes than you have cores will result in wasted resources and cores remaining idle. The correct number of cores to use in your code is determined by the amount of resources you requested to the scheduler. For example, if you have the same computation to perform on many tens of data or more, it would make sense to use all of the cores of a node. In this case, you can write your job submission script with the following header :

and then, your code would become the following : `<tabs>`{=html} `<tab name="Using a loop">`{=html}

`</tab>`{=html} `<tab name="Using a map">`{=html}

`</tab>`{=html} `</tabs>`{=html}

Note that in the above example, the function `cube` itself is sequential. If you are calling some external library, such as `numpy`, it is possible that the functions called by your code are themselves parallel. If you want to distribute processes with the technique above, you should verify whether the functions you call are themselves parallel, and if they are, you need to control how many threads they will take themselves. If, for example, they take all the cores available (32 in the above example), and you are yourself starting 32 processes, this will slow down your code and possibly overload the node as well.

Note that the `multiprocessing` module is restricted to using a single compute node, so the speedup achievable by your program is usually limited to the total number of CPU cores in that node. If you want to go beyond this limit and use multiple nodes, consider using mpi4py or [PySpark](https://docs.alliancecan.ca/Apache_Spark/en#PySpark "PySpark"){.wikilink}. Other methods of parallelizing Python (not all of them necessarily supported on our clusters) are listed [here](https://wiki.python.org/moin/ParallelProcessing). Also note that you can greatly improve the performance of your Python program by ensuring it is written efficiently, so that should be done first before parallelizing. If you are not sure if your Python code is efficient, please contact [technical support](https://docs.alliancecan.ca/technical_support "technical support"){.wikilink} and have them look at your code.

## Anaconda

Please see [Anaconda](https://docs.alliancecan.ca/Anaconda/en "Anaconda"){.wikilink}.

## Jupyter

Please see [Jupyter](https://docs.alliancecan.ca/Jupyter "Jupyter"){.wikilink}.

## Debugging

Debugging your python code might not be obvious. Simple methods such as adding `print` statement or assertion (`assert`) can help you fix some errors.

But often, it is required to dig a bit deeper in the code and its context, using a debugger such as `pdb` is then easier.

You can debug your Python code, in [a small interactive job](https://docs.alliancecan.ca/Running_jobs#Interactive_jobs "a small interactive job"){.wikilink}:

1.  . Add `import pdb; pdb.set_trace()` to the beginning of your file, or add `breakpoint()` at the desired location.
2.  . Run your code: `python ...`
3.  . You\'ll end up in the debugger and can now look around and evaluate expressions.

Useful commands:

  Command         Description
  --------------- ----------------------------------------------------------------------------------------------------------------------------------------------------------
  w (here)        Print a stack trace, with the most recent frame at the bottom. An arrow (\>) indicates the current frame, which determines the context of most commands.
  b (reak)        With a lineno argument, set a break at line lineno in the current file.
  s (tep)         Execute the current line, stop at the first possible occasion (either in a function that is called or on the next line in the current function).
  n (ext)         Continue execution until the next line in the current function is reached or it returns.
  r (eturn)       Continue execution until the current function returns.
  c (ont(inue))   Continue execution, only stop when a breakpoint is encountered.
  p exp           Evaluate expression in the current context and print its value.
  l (ist)         List source code for the current file.
  q (uit)         Quit from the debugger. The program being executed is aborted.

Typically, one would use `w`, `s`, `l`, `p`, `n` to debug a file.

For more information, see [the Python Debugger](https://docs.python.org/3/library/pdb.html).

## Troubleshooting

### Python script is hanging {#python_script_is_hanging}

By using the [faulthandler](https://docs.python.org/3.8/library/faulthandler.html) module, you can edit your script to allow dumping a traceback after a timeout. See `faulthandler.dump_traceback_later()`.

You can also inspect a python process while the job is running, without modifying it beforehand, using [py-spy](https://pythonrepo.com/repo/benfred-py-spy-python-debugging-tools):

1.  Install py-spy in a virtualenv in your home
2.  Attach to the running job, using `srun --pty --jobid JOBID bash`
3.  Use `htop -u $USER` to find the process ID of your python script
4.  Activate the virtualenv where py-spy is installed
5.  Run `py-spy top --pid PID` to see live feedback about where your code is spending time
6.  Run `py-spy dump --pid PID` to get a traceback of where your code is currently at.

=== Package \'X\' requires a different Python: X.Y.Z not in \'\>=X.Y\' === When installing packages, you may encounter an error similar to: `ERROR: Package 'X' requires a different Python: 3.6.10 not in '>=3.7'`.

The current python module loaded (3.6.10 in this case) is not supported by that package. You can update to a more recent version, such as the latest available module. Or install an older version of package \'X\'.

### Package has requirement X, but you\'ll have Y which is incompatible {#package_has_requirement_x_but_youll_have_y_which_is_incompatible}

When installing packages, you may encounter an error similar to: `ERROR: Package has requirement X, but you'll have Y which is incompatible.`.

Upgrade `pip` to the latest version or higher than [`https://pip.pypa.io/en/stable/news/#v21-3 21.3`](https://pip.pypa.io/en/stable/news/#v21-3_21.3 "https://pip.pypa.io/en/stable/news/#v21-3 21.3"){.wikilink} to use the new dependency resolver:

Then rerun your install command.

### No matching distribution found for X {#no_matching_distribution_found_for_x}

When installing packages, you may encounter an error similar to:

`pip` did not find a package to install that satisfies the requirements (name, version or tags). Verify that the name and version are correct. Note also that `manylinux_x_y` wheels are discarded.

You can also verify that the package is available from the wheelhouse with the [avail_wheels](https://docs.alliancecan.ca/Python#Available_wheels "avail_wheels"){.wikilink} command or by searching on [Available Python wheels](https://docs.alliancecan.ca/Available_Python_wheels "Available Python wheels"){.wikilink} page.

### Installing many packages {#installing_many_packages}

When installing multiple packages, it is best to install them in one command when possible:

as this helps `pip` resolve dependencies issues.

### My virtual environment was working yesterday but not anymore {#my_virtual_environment_was_working_yesterday_but_not_anymore}

Packages are often updated and this leads to a non-reproducible virtual environment.

Another reason might be that the virtual environment was created in \$SCRATCH and part of it was deleted with the automatic purge of the filesystem; this would make the virtual environment nonfunctional.

To remedy that, freeze the specific packages and their versions with X.Y\' \'package2X.Y.Z\' \'package3\<X.Y\' \'package4\>X.Y\' }} and then create a [requirements file](https://docs.alliancecan.ca/Python#Creating_virtual_environments_inside_of_your_jobs "requirements file"){.wikilink} that will be used to install the required packages in your job.

### X is not a supported wheel on this platform {#x_is_not_a_supported_wheel_on_this_platform}

When installing a package, you may encounter the following error: `ERROR: package-3.8.1-cp311-cp311-manylinux_2_28_x86_64.whl is not a supported wheel on this platform.`

Some packages may be incompatible or not supported on the systems. Two common cases are:

- trying to install a `manylinux` package
- or a python package built for a different Python version (e.g. installing a package built for python 3.11 when you have python 3.9).

Some `manylinux` package can be made available through the [wheelhouse](https://docs.alliancecan.ca/Available_Python_wheels "wheelhouse"){.wikilink}.

### AttributeError: module 'numpy' has no attribute 'X' {#attributeerror_module_numpy_has_no_attribute_x}

When installing `numpy` without specifying a version number, the latest available version will be installed. In Numpy v1.20, many attributes were set for deprecation and are now [expired in v1.24](https://numpy.org/devdocs/release/1.24.0-notes.html#expired-deprecations).

This may result in an error, depending on the attribute accessed. For example, `AttributeError: module ‘numpy’ has no attribute ‘bool’`.

This can be solved by installing a previous version of Numpy: `pip install --no-index 'numpy<1.24'`.

### ModuleNotFoundError: No module named \'X\' {#modulenotfounderror_no_module_named_x}

When trying to import a Python module, it may not be found. Some common causes are:

- the package is not installed or is not visible to the python interpreter;
- the name of the module to import is not the same as the name of the package that provides it;
- a broken virtual environment.

To avoid such problems, do not:

- modify the `PYTHONPATH` environment variable;
- modify the `PATH` environment variable;
- load a module while a virtual environment is activated (activate your virtual environment only after loading all the required modules)

When you encounter this problem, first make sure you followed the above advice. Then:

- make sure that the package is installed; run `pip list`;
- double-check the module name (upper or lower case and underscores matter);
- make sure that the module is imported at the correct level (when importing from its source directory).

In doubt, start over with a new virtual environment.

### ImportError: numpy.core.multiarray failed to import {#importerror_numpy.core.multiarray_failed_to_import}

When trying to import a Python module that depends on Numpy, one may encounter `ImportError: numpy.core.multiarray failed to import`.

This is caused by an incompatible version of Numpy installed or used and you must install a compatible version.

This is especially true with the [release of Numpy 2.0 which breaks the ABI.](https://numpy.org/devdocs/dev/depending_on_numpy.html#numpy-2-0-specific-advice) In the case of a wheel that was built with version 1.x but installed version 2.x, one must installed a lower version with: `pip install --no-index 'numpy<2.0'`

### Defaulting to user installation because normal site-packages is not writeable {#defaulting_to_user_installation_because_normal_site_packages_is_not_writeable}

When installing packages, one may encounter the message `Defaulting to user installation because normal site-packages is not writeable`.

This is `pip` default behavior outside a virtual environment. This means that no virtual environment was found nor activated and that `pip` tried to install in a location where it does not have permissions to do so.

This results in [local installations](https://docs.alliancecan.ca/Python#Local_installation_(--user) "local installations"){.wikilink} which may be problematic.

### Local installation (\--user) {#local_installation___user}

Local installation can occur unexpectedly (if an error occur with your virtual environment, or permissions issues) or by user defined installation (`pip install --user`).

Local installation is essentially dumping dependencies into one shared space, which is a recipe for headaches. This creates weird import issues or runtime issues with your python packages, or version conflicts which could result in `<i>`{=html}dependency hell`</i>`{=html}.

Using a [virtual environment](https://docs.alliancecan.ca/Python#Creating_virtual_environments_inside_of_your_jobs "virtual environment"){.wikilink} is best for isolation, reproducibility and managing different versions across your different projects.

#### Remove local installation {#remove_local_installation}

To effectively remove local installations, one needs to:

Note that you may need to specify binaries directly if you are using the `~/.local/bin` for local binaries (other than Python packages).

Once the local installations are removed, start over with [ a clean fresh new virtual environment](https://docs.alliancecan.ca/Python#Creating_virtual_environments_inside_of_your_jobs " a clean fresh new virtual environment"){.wikilink}.
