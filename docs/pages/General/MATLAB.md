---
title: "MATLAB/en"
url: "https://docs.alliancecan.ca/wiki/MATLAB/en"
category: "General"
last_modified: "2025-08-29T13:18:02Z"
page_id: 3915
display_title: "MATLAB"
---

`<languages />`{=html}

There are two ways of using MATLAB on our clusters:

`<b>`{=html}1) Running MATLAB directly`</b>`{=html}, but that requires a license. You may either

- run MATLAB on [Fir](https://docs.alliancecan.ca/Fir "Fir"){.wikilink}, [ Narval](https://docs.alliancecan.ca/Narval/en " Narval"){.wikilink} or [Rorqual](https://docs.alliancecan.ca/Rorqual/en "Rorqual"){.wikilink}, all of which have a license available for any student, professor or academic researcher;
- use an external license, i.e., one owned by your institution, faculty, department, or lab. See `<i>`{=html}[Using an external license](https://docs.alliancecan.ca/#Using_an_external_license "Using an external license"){.wikilink}`</i>`{=html} below.

`<b>`{=html}2) Compiling your MATLAB code`</b>`{=html} by using the MATLAB Compiler `mcc` and by running the generated executable file on any cluster. You can use this executable without license considerations.

More details about these approaches are provided below.

# Using an external license {#using_an_external_license}

We are a hosting provider for MATLAB. This means that we have MATLAB installed on our clusters and can allow you to access an external license to run computations on our infrastructure. Arrangements have already been made with several Canadian institutions to make this automatic. To see if you already have access to a license, carry out the following test:

    [name@cluster ~]$ module load matlab/2023b.2
    [name@cluster ~]$ matlab -nojvm -nodisplay -batch license

    987654
    [name@cluster ~]$

If any license number is printed, you\'re okay. Be sure to run this test on each cluster on which you want to use MATLAB, since licenses may not be available everywhere.

If you get the message `<i>`{=html}This version is newer than the version of the license.dat file and/or network license manager on the server machine`</i>`{=html}, try an older version of MATLAB in the `module load` line.

Otherwise, either your institution does not have a MATLAB license, does not allow its use in this way, or no arrangements have yet been made. Find out who administers the MATLAB license at your institution (faculty, department) and contact them or your Mathworks account manager to know if you are allowed to use the license in this way.

If you are allowed, then some technical configuration will be required. Create a file similar to the following example:

Put this file in the `$HOME/.licenses/` directory where the IP address and port number correspond to the values for your campus license server. Next you will need to ensure that the license server on your campus is reachable by our compute nodes. This will require our technical team to get in touch with the technical people managing your license software. Please write to [ technical support](https://docs.alliancecan.ca/Technical_support " technical support"){.wikilink} so that we can arrange this for you.

For online documentation, see <http://www.mathworks.com/support>. For product information, visit <http://www.mathworks.com>.

# Preparing your `.matlab` folder {#preparing_your_.matlab_folder}

Because the /home directory is accessible in read-only mode on some compute nodes, you need to create a `.matlab` symbolic link that makes sure that the MATLAB profile and job data will be written to the /scratch space instead.

    [name@cluster ~]$ cd $HOME
    [name@cluster ~]$ if [ -d ".matlab" ]; then
      mv .matlab scratch/
    else
      mkdir -p scratch/.matlab
    fi && ln -sn scratch/.matlab .matlab

# Available toolboxes {#available_toolboxes}

To see a list of the MATLAB toolboxes available with the license and cluster you\'re using, you can use the following command:

    [name@cluster ~]$  module load matlab
    [name@cluster ~]$  matlab -nojvm -batch "ver"

# Running a serial MATLAB program {#running_a_serial_matlab_program}

`<b>`{=html}Important:`</b>`{=html} Any significant MATLAB calculation (takes more than about 5 minutes or a gigabyte of memory) must be submitted to the scheduler. Here is an example of how to do that. For more on using the scheduler, please see [Running jobs](https://docs.alliancecan.ca/Running_jobs "Running jobs"){.wikilink}.

Consider the following example code:

Here is a Slurm script that you can use to run `cosplot.m`:

Submit the job using `sbatch`:

Each time you run MATLAB, it may create a file like `java.log.12345`. You should delete such files after MATLAB has run so as not to waste storage space. You can also suppress the creation of such files by using the `-nojvm` option, but doing so may interfere with certain plotting functions.

For further information on command line options including `-nodisplay`, `-nojvm`, `-singleCompThread`, `-batch`, and others, see [MATLAB (Linux)](https://www.mathworks.com/help/matlab/ref/matlablinux.html) on the MathWorks website.

# Parallel execution of MATLAB {#parallel_execution_of_matlab}

MATLAB supports a [variety of parallel execution modes](https://www.mathworks.com/help/parallel-computing/quick-start-parallel-computing-in-matlab.html). Most MATLAB users on our clusters will probably find it sufficient to run MATLAB using a `Threads` parallel environment on a single node. Here is an example of how to do that (derived from the [Mathworks documentation for `parfor`](https://www.mathworks.com/help/parallel-computing/parfor.html)):

Save the above MATLAB code in a file called `timeparfor.m`. Then create the following job script and submit it with `sbatch matlab_parallel.sh` to execute the function in parallel using 4 cores:

You may wish to experiment with changing `--cpus-per-task` to other small values (e.g. 1, 2, 6, 8) to observe the effect this has on performance.

## Simultaneous parallel MATLAB jobs {#simultaneous_parallel_matlab_jobs}

If you are using a `Cluster` parallel environment as described [here](https://www.mathworks.com/help/parallel-computing/quick-start-parallel-computing-in-matlab.html#mw_d4204011-7467-47d9-b765-33dc8a8f83cd), the following problem may arise. When two or more parallel MATLAB jobs call `parpool` at the same time, the different jobs try to read and write to the same `.dat` file in the `$HOME/.matlab/local_cluster_jobs/R*` folder, which corrupts the local parallel profile used by other MATLAB jobs. If this has occurred to you, delete the `local_cluster_jobs` folder when no job is running.

To avoid this problem we recommend that you ensure each job creates its own parallel profile in a unique location by setting the `JobStorageLocation` property of the [`parallel.Cluster`](https://www.mathworks.com/help/parallel-computing/parallel.cluster.html) object, as shown in the following code fragment:

References:

- FAS Research Computing, [`<i>`{=html}MATLAB Parallel Computing Toolbox simultaneous job problem`</i>`{=html}](https://www.rc.fas.harvard.edu/resources/documentation/software/matlab-pct-simultaneous-job-problem/).
- MathWorks, [`<i>`{=html}Why am I unable to start a local MATLABPOOL from multiple MATLAB sessions that use a shared preference directory using Parallel Computing Toolbox 4.0 (R2008b)?`</i>`{=html}](https://www.mathworks.com/matlabcentral/answers/97141-why-am-i-unable-to-start-a-local-matlabpool-from-multiple-matlab-sessions-that-use-a-shared-preferen)

# Using the Compiler and Runtime libraries {#using_the_compiler_and_runtime_libraries}

`<b>`{=html}Important:`</b>`{=html} Like any other intensive job, you must always run MCR code within a job submitted to the scheduler. For instructions on using the scheduler, please see the [Running jobs](https://docs.alliancecan.ca/Running_jobs "Running jobs"){.wikilink} page.

You can also compile your code using MATLAB Compiler, which is included among the modules we host. See documentation for the compiler on the [MathWorks](https://www.mathworks.com/help/compiler/index.html) website. At the moment, mcc is provided for versions 2014a, 2018a and later.

To compile the `cosplot.m` example given above, you would use the command

This will produce a binary named `cosplot`, as well as a wrapper script. To run the binary on our servers, you will only need the binary. The wrapper script named `run_cosplot.sh` will not work as is on our servers because MATLAB assumes that some libraries can be found in specific locations. Instead, we provide a different wrapper script called `run_mcr_binary.sh` which sets the correct paths.

On one of our servers, load an MCR [module](https://docs.alliancecan.ca/Utiliser_des_modules/en "module"){.wikilink} corresponding to the MATLAB version you used to build the executable:

Run the following command:

then, in your submission script (`<b>`{=html}not on the login nodes`</b>`{=html}), use your binary as so: `run_mcr_binary.sh cosplot`

You will only need to run the `setrpaths.sh` command once for each compiled binary. The `run_mcr_binary.sh` will instruct you to run it if it detects that it has not been done.

# Using the MATLAB Parallel Server {#using_the_matlab_parallel_server}

MATLAB Parallel Server is only worthwhile `<b>`{=html}if you need more workers in your parallel MATLAB job than available CPU cores on a single compute node`</b>`{=html}. While a regular MATLAB installation (see above sections) allows you to run parallel jobs within one node (up to 64 workers per job, depending on which node and cluster), the MATLAB Parallel Server is the licensed MathWorks solution for running a parallel job on more than one node.

This solution usually works by submitting MATLAB parallel jobs from a local MATLAB interface on your computer. `<b>`{=html}Since May 2023, some mandatory security improvements have been implemented on all clusters. Because MATLAB uses an SSH mode that is no longer permitted, job submission from a local computer is no longer possible until MATLAB uses a new connection method. There is currently no workaround.`</b>`{=html}

## Slurm plugin for MATLAB {#slurm_plugin_for_matlab}

`<b>`{=html}The procedure below no longer works because the Slurm plugin is no longer available and because of the SSH issue described above.`</b>`{=html} The configuration steps are kept until a workaround is found:

1.  Have MATLAB R2022a or newer installed, `<b>`{=html}including the Parallel Computing Toolbox`</b>`{=html}.
2.  Go to the MathWorks Slurm Plugin page, `<b>`{=html}download and run`</b>`{=html} the `*.mlpkginstall` file. (i.e., click on the blue `<i>`{=html}Download`</i>`{=html} button on the right side, just above the `<i>`{=html}Overview`</i>`{=html} tab.)
3.  Enter your MathWorks credentials; if the configuration wizard does not start, run in MATLAB

    :   `parallel.cluster.generic.runProfileWizard()`
4.  Give these responses to the configuration wizard:
    - Select `<b>`{=html}Unix`</b>`{=html} (which is usually the only choice)
    - Shared location: `<b>`{=html}No`</b>`{=html}
    - Cluster host:
      - For Narval: `<b>`{=html}narval.alliancecan.ca`</b>`{=html}
      - For Rorqual: `<b>`{=html}rorqual.alliancecan.ca`</b>`{=html}
    - Username (optional): Enter your Alliance username (the identity file can be set later if needed)
    - Remote job storage: `<b>`{=html}/scratch`</b>`{=html}
      - Keep `<i>`{=html}Use unique subfolders`</i>`{=html} checked
    - Maximum number of workers: `<b>`{=html}960`</b>`{=html}
    - Matlab installation folder for workers (both local and remote versions must match):
      - For local R2022a: `<b>`{=html}/cvmfs/restricted.computecanada.ca/easybuild/software/2020/Core/matlab/2022a`</b>`{=html}
    - License type: `<b>`{=html}Network license manager`</b>`{=html}
    - Profile Name: `<b>`{=html}narval`</b>`{=html} or `<b>`{=html}rorqual`</b>`{=html}
5.  Click on `<i>`{=html}Create`</i>`{=html} and `<i>`{=html}Finish`</i>`{=html} to finalize the profile.

## Edit the plugin once installed {#edit_the_plugin_once_installed}

In MATLAB, go to the `nonshared` folder (i.e., run the following in the MATLAB terminal):

`cd(fullfile(matlabshared.supportpkg.getSupportPackageRoot, 'parallel', 'slurm', 'nonshared'))`

Then:

1.  Open the `<b>`{=html}independentSubmitFcn.m`</b>`{=html} file; around line #117 is the line
    `additionalSubmitArgs = sprintf('--ntasks=1 --cpus-per-task=%d', cluster.NumThreads);`

    Replace this line with

    `additionalSubmitArgs = ccSBATCH().getSubmitArgs();`
2.  Open the `<b>`{=html}communicatingSubmitFcn.m`</b>`{=html} file; around line #126 is the line
    `additionalSubmitArgs = sprintf('--ntasks=%d --cpus-per-task=%d', environmentProperties.NumberOfTasks, cluster.NumThreads);`

    Replace this line with

    `additionalSubmitArgs = ccSBATCH().getSubmitArgs();`
3.  Open the `<b>`{=html}communicatingJobWrapper.sh`</b>`{=html} file; around line #20 (after the copyright statement), add the following command and adjust the module version to your local Matlab version:
    `module load matlab/2022a`

Restart MATLAB and go back to your home directory:

`cd(getenv('HOME'))  # or cd(getenv('HOMEPATH')) on Windows`

## Validation

`<b>`{=html}Do not`</b>`{=html} use the built-in validation tool in the `<i>`{=html}Cluster Profile Manager`</i>`{=html}. Instead, you should try the `TestParfor` example, along with a proper `ccSBATCH.m` script file:

1.  Download and extract code samples on GitHub at <https://github.com/ComputeCanada/matlab-parallel-server-samples>.
2.  In MATLAB, go to the newly extracted `TestParfor` directory.
3.  Follow instructions in <https://github.com/ComputeCanada/matlab-parallel-server-samples/blob/master/README.md>.

Note: When the `ccSBATCH.m` is in your current working directory, you may try the `<i>`{=html}Cluster Profile Manager`</i>`{=html} validation tool, but only the first two tests will work. Other tests are not yet supported.

# External resources {#external_resources}

MathWorks provides a variety of documentation and training about MATLAB.

- See [<https://www.mathworks.com/help/matlab/>](https://www.mathworks.com/help/matlab/) for documentation (many languages)
- See [<https://matlabacademy.mathworks.com/>](https://matlabacademy.mathworks.com/) for self-paced online courses (EN, JP, ES, KR, CN)

Some universities also provide their own MATLAB documentation:

- More examples with job scripts: [<https://rcs.ucalgary.ca/MATLAB>](https://rcs.ucalgary.ca/MATLAB)
