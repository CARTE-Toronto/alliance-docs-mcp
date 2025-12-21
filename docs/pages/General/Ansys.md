---
title: "Ansys"
url: "https://docs.alliancecan.ca/wiki/Ansys"
category: "General"
last_modified: "2025-12-15T15:01:17Z"
page_id: 4568
display_title: "Ansys"
---

Ansys is a software suite for engineering simulation and 3-D design. It includes packages such as Ansys Fluent and Ansys CFX.

= Licensing =
We are a hosting provider for Ansys. This means that we have the software installed on our clusters, but we do not provide a generic license accessible to everyone. However, many institutions, faculties, and departments already have licenses that can be used on our clusters.  Once the legal aspects are worked out for licensing, there will be remaining technical aspects. The license server on your end will need to be reachable by our compute nodes. This will require our technical team to get in touch with the technical people managing your license software. In some cases, this has already been done. You should then be able to load the Ansys module, and it should find its license automatically. If this is not the case, please contact our technical support so that they can arrange this for you.

== Configuring your license file ==
Our module for Ansys is designed to look for license information in a few places. One of those places is your /home folder. You can specify your license server by creating a file named $HOME/.licenses/ansys.lic consisting of two lines as shown.  Customize the file by replacing FLEXPORT and LICSERVER with the appropriate values for your server.

 FILE: ansys.lic

setenv("ANSYSLMD_LICENSE_FILE", "FLEXPORT@LICSERVER")

The following table provides established values for the CMC and SHARCNET license servers.  To use a different server, locate the corresponding values as explained in Local license servers.

 TABLE: Preconfigured license servers

License 	System/Cluster	LICSERVER                	FLEXPORT	NOTES
CMC     	fir           	172.26.0.101             	6624    	None
CMC     	narval/rorqual	10.100.64.10             	6624    	None
CMC     	nibi          	10.25.1.56               	6624    	NewIP Feb21/2025
CMC     	trillium      	scinet-cmc               	6624    	None
SHARCNET	fir           	license1.computecanada.ca	1055    	Currently NOT working
SHARCNET	narval/rorqual	license3.sharcnet.ca     	1055    	Supports <= ansys/2024R2.04
SHARCNET	nibi          	license1.computecanada.ca	1055    	Supports <= ansys/2025R1.02
SHARCNET	trillium      	localhost                	1055    	Currently NOT working

Researchers who purchase a new CMC license subscription must submit your Alliance account username otherwise license checkouts will fail. The number of cores that can be used with a CMC license is described in the Other Tricks and Tips sections of the Ansys Electronics Desktop and  Ansys Mechanical/Fluids quick start guides.

=== Local license servers  ===

Before a local institutional Ansys license server can be used on the Alliance, firewall changes will need to be done on both the server and cluster side.  For many Ansys servers this work has already been done and they can be used by following the steps in the "Ready To Use" section below.  For Ansys servers that have never used on the Alliance, two additional steps must be done as shown in the "Setup Required" section also below.

==== Ready to use ====

To use a local institutional ANSYS License server whose network/firewall connections have already been setup for use on an Alliance cluster, contact your local Ansys license server administrator and get the following two pieces of information:
 1) the configured Ansys flex port (FLEXPORT) number commonly 1055
 2) the fully qualified hostname (LICSERVER) of the license server
Then simply configure your ~/.licenses/ansys.lic file by plugging the values for FLEXPORT and LICSERVER into the FILE: ansys.lic template above.

==== Setup required ====

To use a local Ansys license server that has never been setup for use with an Alliance cluster, then you will ALSO need to get the following from your local Ansys license server administrator:
  3) the statically configured vendor port (VENDPORT) of the license server
  4) confirmation  will resolve to the same IP address as LICSERVER on the cluster
where the  can be found in the first line of the license file with format "SERVER   ".   Send items 1->3 by email to technical support and mention which Alliance cluster you want to run Ansys jobs on.  An Alliance system administrator will then open the outbound cluster firewall so license checkout requests can reach your license server from the cluster compute nodes.  A range of IP addresses will then be sent back to you.  Give these to your local network administrator.  Request the firewall into your local license server be opened so that Ansys license connection (checkout requests) can reach your servers FLEXPORT and VENDPORT ports across the IP range.

== Checking license ==

To test if your ansys.lic is configured and working properly copy/paste the following sequence of commands on the cluster you are submitting jobs to.  The only required change would be to specify YOURUSERID.  If the software on a remote license server has not been updated then a failure can occur if the latest module version of ansys is loaded to test with.  Therefore to be certain the license checkouts will work when jobs are run in the queue, the same ansys module version that you load in your slurm scripts should be specified below.
 [login-node:~] cd /tmp
 [login-node:~] salloc --time1:0:0 --mem1000M --accountdef-YOURUSERID
 [login-node:~] module load StdEnv/2023; module load ansys/2023R2
 [login-node:~] $EBROOTANSYS/v$(echo ${EBVERSIONANSYS:2:2}${EBVERSIONANSYS:5:1})/licensingclient/linx64/lmutil lmstat -c $ANSYSLMD_LICENSE_FILE | grep "ansyslmd: UP" 1> /dev/null && echo Success  echo Fail

If Success is output license checkouts should work when jobs are submitted to the queue.
If Fail is output then jobs will likely fail requiring a problem ticket to be submitted to resolve.

= Version compatibility =

Ansys simulations are typically forward compatible but NOT backwards compatible.  This means that simulations created using an older version of Ansys can be expected to load and run fine with any newer version.  For example, a simulation created and saved with ansys/2022R2 should load and run smoothly with ansys/2023R2 but NOT the other way around.  While it may be possible to start a simulation running with an older version random error messages or crashing will likely occur.  Regarding Fluent simulations, if you cannot recall which version of ansys was used to create your cas file try grepping it as follows to look for clues :

$ grep -ia fluent combustor.cas
   (0 "fluent15.0.7  build-id: 596")

$ grep -ia fluent cavity.cas.h5
   ANSYS_FLUENT 24.1 Build 1018

== Platform support ==

Ansys provides detailed platform support information describing software/hardware compatibility for the Current Release and Previous Releases.   The Platform Support by Application / Product pdf is of special interest since it shows which packages are supported under Windows but not under Linux and thus not on the Alliance such as Spaceclaim.

== What's new ==

Ansys posts Product Release and Updates for the latest releases.  Similar information for previous releases can generally be pulled up for various application topics by visiting the Ansys blog page and using the FILTERS search bar.  For example, searching on What’s New Fluent 2024 gpu pulls up a document with title What’s New for Ansys Fluent in 2024 R1? containing a wealth of the latest gpu support information. Specifying a version number in the Press Release search bar is also a good way to find new release information.   Recently a module for the latest ANSYS release was installed  ansys/2025R1.02 however to use it requires a suitably updated license server such as CMCs.  The upgrade of the SHARCNET license server is underway however until it is complete (and this message updated accordingly) it will only support jobs run with  ansys/2024R2.04 or older. To request a new version be installed or a problem with an exiting module please submit a ticket.

== Service packs ==

Starting with Ansys 2024 a separate Ansys module will appear on the clusters with a decimal and two digits appearing after the release number whenever a service pack is been installed over the initial release.  For example, the initial 2024 release with no service pack applied may be loaded with  module load ansys/2024R1 while a module with Service Pack 3 applied may be loaded with module load ansys/2024R1.03 instead.  If a service pack is already available by the time a new release is to be installed, then most likely only a module for that service pack number will be installed unless a request to install the initial release is also received.

Most users will likely want to load the latest module version equipped with the latest installed service pack which can be achieved by simply doing module load ansys.  While it's not expected service packs will impact numerical results, the changes they make are extensive and so, if computations have already been done with the initial release or an earlier service pack then some groups may prefer to continue using it. Having separate modules for each service pack makes this possible.  Starting with Ansys 2024R1 a detailed description of what each service pack does can be found by searching this link for Service Pack Details.  Future versions will presumably be similarly searchable by manually modifying the version number contained in the link.

= Cluster batch job submission =
The Ansys software suite comes with multiple implementations of MPI to support parallel computation. Unfortunately, none of them support our Slurm scheduler. For this reason, we need special instructions for each Ansys package on how to start a parallel job. In the sections below, we give examples of submission scripts for some of the packages.  While the slurm scripts should work with on all clusters, Niagara users may need to make some additional changes covered here.

== Ansys Fluent ==
Typically, you would use the following procedure to run Fluent on one of our clusters:

# Prepare your Fluent job using Fluent from the Ansys Workbench on your desktop machine up to the point where you would run the calculation.
# Export the "case" file with File > Export > Case… or find the folder where Fluent saves your project's files. The case file will often have a name like FFF-1.cas.gz.
# If you already have data from a previous calculation, which you want to continue, export a "data" file as well (File > Export > Data…) or find it in the same project folder (FFF-1.dat.gz).
# Transfer the case file (and if needed the data file) to a directory on the /project or /scratch filesystem on the cluster.  When exporting, you can save the file(s) under a more instructive name than FFF-1.* or rename them when they are uploaded.
# Now you need to create a "journal" file. Its purpose is to load the case file (and optionally the data file), run the solver and finally write the results.  See examples below and remember to adjust the filenames and desired number of iterations.
# If jobs frequently fail to start due to license shortages and manual resubmission of failed jobs is not convenient, consider modifying your script to requeue your job (up to 4 times) as shown under the by node + requeue tab further below.  Be aware that doing this will also requeue simulations that fail due to non-license related issues (such as divergence), resulting in lost compute time.  Therefore it is strongly recommended to monitor and inspect each Slurm output file to confirm each requeue attempt is license related.  When it is determined that a job is requeued due to a simulation issue, immediately manually kill the job progression with scancel jobid and correct the problem.
# After running the job, you can download the data file and import it back into Fluent with File > Import > Data….

=== Slurm scripts ===

==== General purpose ====

Most Fluent jobs should use the following by node script to minimize solution latency and maximize performance over as few nodes as possible. Very large jobs, however, might wait less in the queue if they use a by core script. However, the startup time of a job using many nodes can be significantly longer, thus offsetting some of the benefits. In addition, be aware that running large jobs over an unspecified number of potentially very many nodes will make them far more vulnerable to crashing if any of the compute nodes fail during the simulation. The scripts will ensure Fluent uses shared memory for communication when run on a single node or distributed memory (utilizing MPI and the appropriate HPC interconnect) when run over multiple nodes.  The two narval tabs may be be useful to provide a more robust alternative if fluent crashes during the initial auto mesh partitioning phase when using the standard intel based scripts with the parallel solver.  The other option would be to manually perform the mesh partitioning in the fluent gui then try to run the job again on the cluster with the intel scripts.  Doing so will allow you to inspect the partition statistics and specify the partitioning method to obtain an optimal result.  The number of mesh partitions should be an integral multiple of the number of cores; for optimal efficiency, ensure at least 10000 cells per core.

 uniq`; do echo "${i}:$(cat /tmp/mf-$SLURM_JOB_ID  grep $i  wc -l)" >> /tmp/machinefile-$SLURM_JOB_ID; done
NCORES=$SLURM_NTASKS

if [ "$SLURM_NNODES" == 1 ]; then
 fluent -g $MYVERSION -t $NCORES -mpi=openmpi -pshmem -i $MYJOURNALFILE
else
 export FI_PROVIDER=verbs
 fluent -g $MYVERSION -t $NCORES -mpi=openmpi -pib -cnf=/tmp/machinefile-$SLURM_JOB_ID -i $MYJOURNALFILE
fi
}}

 uniq`; do echo "${i}:$(cat /tmp/mf-$SLURM_JOB_ID  grep $i  wc -l)" >> /tmp/machinefile-$SLURM_JOB_ID; done
NCORES=$SLURM_NTASKS

if [ "$SLURM_NNODES" == 1 ]; then
 fluent -g $MYVERSION -t $NCORES -mpi=openmpi -pshmem -i $MYJOURNALFILE
else
 export FI_PROVIDER=verbs
 fluent -g $MYVERSION -t $NCORES -mpi=openmpi -pib -cnf=/tmp/machinefile-$SLURM_JOB_ID -i $MYJOURNALFILE
fi
}}

==== License requeue ====

The scripts in this section should only be used with Fluent jobs that are known to complete normally without generating any errors in the output however typically require multiple requeue attempts to checkout licenses.  They are not recommended for Fluent jobs that may 1) run for a long time before crashing 2) run to completion but contain unresolved journal file warnings, since in both cases the simulations will be repeated from the beginning until the maximum number of requeue attempts specified by the array value is reached.  For these types of jobs, the general purpose scripts above should be used instead.

==== Solution restart ====

The following two scripts are provided to automate restarting very large jobs that require more than the typical seven-day maximum runtime window available on most clusters. Jobs are restarted from the most recent saved time step files. A fundamental requirement is the first time step can be completed within the requested job array time limit (specified at the top of your Slurm script) when starting a simulation from an initialized solution field. It is assumed that a standard fixed time step size is being used. To begin, a working set of sample.cas, sample.dat and sample.jou files must be present. Next edit your sample.jou file to contain /solve/dual-time-iterate 1 and /file/auto-save/data-frequency 1. Then create a restart journal file by doing cp sample.jou sample-restart.jou and edit the sample-restart.jou file to contain /file/read-cas-data sample-restart instead of /file/read-cas-data sample and comment out the initialization line with a semicolon for instance ;/solve/initialize/initialize-flow. If your 2nd and subsequent time steps are known to run twice as fast (as the initial time step), edit sample-restart.jou to specify /solve/dual-time-iterate 2. By doing this, the solution will only be restarted after two 2 time steps are completed following the initial time step. An output file for each time step will still be saved in the output subdirectory. The value 2 is arbitrary but should be chosen such that the time for 2 steps fits within the job array time limit. Doing this will minimize the number of solution restarts which are computationally expensive. If your first time step performed by sample.jou starts from a converged (previous) solution, choose 1 instead of 2 since likely all time steps will require a similar amount of wall time to complete. Assuming 2 is chosen, the total time of simulation to be completed will be 1*Dt+2*Nrestart*Dt where Nrestart is the number of solution restarts specified in the script. The total number of time steps (and hence the number of output files generated) will therefore be 1+2*Nrestart. The value for the time resource request should be chosen so the initial time step and subsequent time steps will complete comfortably within the Slurm time window specifiable up to a maximum of "#SBATCH --time=07-00:00" days.

=== Journal files ===

Fluent journal files can include basically any command from Fluent's Text-User-Interface (TUI); commands can be used to change simulation parameters like temperature, pressure and flow speed. With this you can run a series of simulations under different conditions with a single case file, by only changing the parameters in the journal file. Refer to the Fluent User's Guide for more information and a list of all commands that can be used.  The following journal files are set up with /file/cff-files no to use the legacy .cas/.dat file format (the default in module versions 2019R3 or older).  Set this instead to /file/cff-files yes to use the more efficient .cas.h5/.dat.h5 file format (the default in module versions 2020R1 or newer).

=== UDFs ===

The first step is to transfer your User-Defined Function or UDF (namely the sampleudf.c source file and any additional dependency files) to the cluster.  When uploading from a windows machine, be sure the text mode setting of your transfer client is used otherwise fluent won't be able to read the file properly on the cluster since it runs linux.  The UDF should be placed in the directory where your journal, cas and dat files reside.  Next add one of the following commands into your journal file before the commands that read in your simulation cas/dat files.   Regardless of whether you use the Interpreted or Compiled UDF approach,  before uploading your cas file onto the Alliance please check that neither the Interpreted UDFs Dialog Box or the UDF Library Manager Dialog Box are configured to use any UDF; this will ensure that only the journal file commands are in control when jobs are submitted.

==== Interpreted ====

To tell fluent to interpret your UDF at runtime, add the following command line into your journal file before the cas/dat files are read or initialized. The filename sampleudf.c should be replaced with the name of your source file.  The command remains the same regardless if the simulation is being run in serial or parallel.  To ensure the UDF can be found in the same directory as the journal file, open your cas file in the fluent gui, remove any managed definitions and resave it.   Doing this will ensure only the following command/method is in control when fluent runs. To use an interpreted UDF with parallel jobs, it will need to be parallelized as described in the section below.

define/user-defined/interpreted-functions "sampleudf.c" "cpp" 10000 no

==== Compiled ====

To use this approach, your UDF must be compiled on an Alliance cluster at least once.  Doing so will create a libudf subdirectory structure containing the required libudf.so shared library.   The libudf directory cannot simply be copied from a remote system (such as your laptop) to the Alliance since the library dependencies of the shared library will not be satisfied, resulting in fluent crashing on startup.  That said, once you have compiled your UDF on an Alliance cluster, you can transfer the newly created libudf to any other Alliance cluster, providing your account loads the same StdEnv environment module version.  Once copied, the UDF can be used by uncommenting the second (load) libudf line below in your journal file when submitting jobs to the cluster.  Both (compile and load) libudf lines should not be left uncommented in your journal file when submitting jobs on the cluster, otherwise your UDF will automatically (re)compiled for each and every job.  Not only is this highly inefficient, but it will also lead to racetime-like build conflicts if multiple jobs are run from the same directory. Besides configuring your journal file to build your UDF, the fluent gui (run on any cluster compute node or gra-vdi) may also be used.  To do this, you would navigate to the Compiled UDFs Dialog Box, add the UDF source file and click Build.   When using a compiled UDF with parallel jobs, your source file should be parallelized as discussed in the section below.

define/user-defined/compiled-functions compile libudf yes sampleudf.c "" ""

and/or

define/user-defined/compiled-functions load libudf

==== Parallel ====

Before a UDF can be used with a fluent parallel job (single node SMP and multinode MPI) it will need to be parallelized.  By doing this we control how/which processes (host and/or compute) run specific parts of the UDF code when fluent is run in parallel on the cluster. The instrumenting procedure involves adding compiler directives, predicates and reduction macros into your working serial UDF. Failure to do so will result in fluent running slow at best or immediately crashing at worst.  The end result will be a single UDF that runs efficiently when fluent is used in both serial and parallel mode.  The subject is described in detail under Part I: Chapter 7: Parallel Considerations of the Ansys 2024 Fluent Customization Manual which can be accessed here.

==== DPM ====
UDFs can be used to customize Discrete Phase Models (DPM) as described in Part III: Solution Mode | Chapter 24: Modeling Discrete Phase | 24.2 Steps for Using the Discrete Phase Models| 24.2.6 User-Defined Functions of the 2024R2 Fluent Users Guide and section Part I: Creating and Using User Defined Functions | Chapter 2: DEFINE Macros | 2.5 Discrete Phase Model (DPM) DEFINE Macros of the 2024R2 Fluent Customization Manual. Before a DMP based UDF can be worked into a simulation, the injection of a set of particles must be defined by specifying "Point Properties" with variables such as source position, initial trajectory, mass flow rate, time duration, temperature and so forth depending on the injection type.  This can be done in the gui by clicking the Physics panel, Discrete Phase to open the Discrete Phase Model box and then clicking the Injections button.  Doing so will open an Injections dialog box where one or more injections can be created by clicking the Create button.   The "Set Injection Properties" dialog which appears will contain an "Injection Type" pulldown with first four types available are "single, group, surface, flat-fan-atomizer". If you select any of these then you can then the "Point Properties" tab can be selected to input the corresponding Value fields.  Another way to specify the "Point Properties" would be to read an injection text file.  To do this select "file" from the Injection Type pulldown, specify the Injection Name to be created and then click the File button (located beside the OK button at the bottom of the  "Set Injection Properties" dialog).   Here either an Injection Sample File (with .dpm extension) or a manually created injection text file can be selected.   To Select the File in the Select File dialog box that change the File of type pull down to All Files (*), then highlight the file which could have any arbitrary name but commonly does have a .inj extension, click the OK button.   Assuming there are no problems with the file, no Console error or warning message will appear in fluent.   As you will be returned to the "Injections" dialog box, you should see the same Injection name that you specified in the "Set Injection Properties" dialog and be able to List its Particles and Properties in the console.  Next open the Discrete Phase Model Dialog Box and select Interaction with Continuous Phase which will enable updating DPM source terms every flow iteration.  This setting can be saved in your cas file or added via the journal file as shown.  Once the injection is confirmed working in the gui the steps can be automated by adding commands to the journal file after solution initialization, for example:
 /define/models/dpm/interaction/coupled-calculations yes
 /define/models/dpm/injections/delete-injection injection-0:1
 /define/models/dpm/injections/create injection-0:1 no yes file no zinjection01.inj no no no no
 /define/models/dpm/injections/list-particles injection-0:1
 /define/models/dpm/injections/list-injection-properties injection-0:1
where a basic manually created injection steady file format might look like:
  $ cat  zinjection01.inj
  (z=4 12)
  ( x          y        z    u         v    w    diameter  t         mass-flow  mass  frequency  time name )
  (( 2.90e-02  5.00e-03 0.0 -1.00e-03  0.0  0.0  1.00e-04  2.93e+02  1.00e-06   0.0   0.0        0.0 ) injection-0:1 )
noting that injection files for DPM simulations are generally setup for either steady or unsteady particle tracking where the format of the former is described in subsection Part III: Solution Mode | Chapter 24: Modeling Discrete Phase | 24.3. Setting Initial Conditions for the Discrete Phase | 24.3.13 Point Properties for File Injections | 24.3.13.1 Steady File Format of the 2024R2 Fluent Customization Manual.

== Ansys CFX ==

=== Slurm scripts ===

A summary of command-line options can be printed by running cfx5solve -help where the same module version thats loaded in your slurm script should be first manually loaded.  By default cfx5solve will run in single precision (-single).  To run cfx5solve in double precision add the -double option noting that doing so will also double memory requirements.  By default cfx5solve can support meshes with up to 80 million elements (structured) or 200 million elements (unstructured).  For larger meshes with up to 2 billion elements, add the -large option.  Various combinations of these options can be specified for the Partitioner, Interpolator or Solver.  Consult the ANSYS CFX-Solver Manager User's Guide for further details.

== Workbench ==

Before submitting a project file to the queue on a cluster (for the first time) follow these steps to initialize it.
# Connect to the cluster with TigerVNC.
# Switch to the directory where the project file is located (YOURPROJECT.wbpj) and start Workbench with the same Ansys module you used to create your project.
# In Workbench, open the project with File -> Open.
# In the main window, right-click on Setup and select Clear All Generated Data.
# In the top menu bar pulldown, select File -> Exit to exit Workbench.
# In the Ansys Workbench popup, when asked The current project has been modified. Do you want to save it?, click on the No button.
# Quit Workbench and submit your job using one of the Slurm scripts shown below.

To avoid writing the solution when a running job successfully completes remove ;Save(Overwrite=True) from within the last line of your script.  Doing this will make it easier to run multiple test jobs (for scaling purposes when changing ntasks), since the initialized solution will not be overwritten each time.  Alternatively, keep a copy of the initialized YOURPROJECT.wbpj file and YOURPROJECT_files subdirectory and restore them after the solution is written.

=== Slurm scripts ===

A project file can be submitted to the queue by customizing one of the following scripts and then running the sbatch script-wbpj-202X.sh command:

== Mechanical ==

The input file can be generated from within your interactive Workbench Mechanical session by clicking Solution -> Tools -> Write Input Files then specify File name: YOURAPDLFILE.inp and Save as type: APDL Input Files (*.inp).  APDL jobs can then be submitted to the queue by running the sbatch script-name.sh command.

=== Slurm scripts ===

In the following slurm scripts, lines beginning with ##SBATCH are commented.

Ansys allocates 1024 MB total memory and 1024 MB database memory by default for APDL jobs. These values can be manually specified (or changed) by adding arguments -m 1024 and/or -db 1024 to the mapdl command line in the above scripts. When using a remote institutional license server with multiple Ansys licenses, it may be necessary to add -p aa_r or -ppf anshpc, depending on which Ansys module you are using. As always, perform detailed scaling tests before running production jobs to ensure that the optimal number of cores and minimum amount memory is specified in your scripts. The single node (SMP Shared Memory Parallel) scripts will typically perform better than the multinode (DIS Distributed Memory Parallel) scripts and therefore should be used whenever possible. To help avoid compatibility issues the Ansys module loaded in your script should ideally match the version used to generate the input file:

 [gra-login2:~/testcase] cat YOURAPDLFILE.inp | grep version
 ! ANSYS input file written by Workbench version 2019 R3

== Ansys ROCKY ==

Besides being able to run simulations in gui mode (as discussed in the Graphical usage section below) Ansys Rocky can also run simulations in non-gui mode.  Both modes support running Rocky with cpus only or with cpus and gpus.  In the below section two sample slurm scripts are  provided where each script would be submitted to the graham queue with the sbatch command as per usual.  At the time of this writing neither script has been tested and therefore extensive customization will likely be required.  It's important to note that these scripts are only usable on graham since the rocky module which they both load is only (at the present time) installed on graham (locally).

=== Slurm scripts ===

To get a full listing of command line options run Rocky -h on the command line after loading any rocky module (currently only ansysrocky/2023R2 is available on Graham).   If Rocky is being run with gpus to solving coupled problems, the number of cpus you should request from slurm (on the same node) should be increased to a maximum until the scalability limit of the coupled application is reached.   If however Rocky is being run with gpus to solve standalone uncoupled problems, then only a minimal number of cpus should be requested that will allow be sufficient for Rocky to still run optimally.  For instance only 2cpus or possibly 3cpus may be required.  When Rocky is run with >= 4 cpus then rocky_hpc licenses will be required which the SHARCNET license does provide.

= Graphical use =

To run Ansys programs in graphical mode click on one of the following OnDemand or Jupyterhub links.  A job submission web page to configure the resources for an interactive session should appear in your browser :

NIBI: https://ondemand.sharcnet.ca
 FIR: https://jupyterhub.fir.alliancecan.ca
 RORQUAL: https://jupyterhub.rorqual.alliancecan.ca
 NARVAL:  https://jupyterhub.narval.alliancecan.ca/
 TRILLIUM: https://ondemand.scinet.utoronto.ca

Submit your resource request and then wait.  If you started a Juypter Lab launcher interface then you can simply load an ansys software module from the left side menu and then click one of the ansys icons to start cfx, fluent mapdl or workbench.  Otherwise if you started a Compute/Basic Desktop from the Nibi OnDemand system then you will need to open a terminal window and manually load an ansys module and run one of the following programs from the command line.  For this later case, if your work requires accelerated graphics then either a whole GPU resource (H100 or T4 at the time of this writing) should be requested.  Since the various ansys applications launched in graphical mode behave differently when different ansys module versions are loaded, recommendations for adding command line argument and exporting environment variables for virtualgl or mesa environments have been documented below depending on whether a GPU has been requested or not and wether a On Demand or Juypter Lab system launcher is being used.

=== Fluids ===

==== Fluent ====

When starting Fluent from a terminal window command line of an On Demand Desktop or the convenience Icon of a Juypter Lab Desktop the following steps should be done including setting the indicated Environment Variables depending on which type of Compute Node the desktop is being started on (with or without a gpu).

::: module load StdEnv/2023 ansys/2025R1
::: fluent

Compute Node (no GPU requested) or Basic Desktop

::: In the Fluent Launcher Click the Environment Tab
::: Copy/paste the following environment variable settings:
:::: export I_MPI_HYDRA_BOOTSTRAP=ssh     (required on nibi)
:::: HOOPS_PICTURE=opengl2-mesa           (2025R1 or newer)
:::: HOOPS_PICTURE=null                   (2024R2 or older)
::: Click the Start button

Compute Node (with GPU requested)

::: In the Fluent Launcher Click the Environment Tab
::: Copy/paste the following environment variable settings:
:::: I_MPI_HYDRA_BOOTSTRAP=ssh            (required on nibi)
:::: HOOPS_PICTURE=opengl2                (2025R1 or newer)
:::: HOOPS_PICTURE=opengl                 (2024R2 or older)
::: Click the Start button

==== CFX ====

When starting CFX from an On Demand Desktop the following arguments maybe specified on the terminal window command line depending on whether a GPU was requested when the Desktop was started.

::: module load StdEnv/2023 ansys/2025R1  (or older)
::: cfx5 -graphics mesa   (no GPU requested)
::: cfx5 -graphics ogl    (with GPU requested)

=== Mapdl ===

The following steps for starting the Mechanical APDL gui from the command line of a terminal window should work regardless if you have started your On Demand Desktop on a Compute node with or without a gpu.

::: module load StdEnv/2023 ansys/2022R2 (or newer versions)
::: mapdl -g, or,
::: launcher then click RUN button

=== Workbench ===

Note that when starting Fluent from within Workbench the same GPU dependent Environment Variable settings should be specified in the Environment Tabl of the Fluent Launcher that are explained in the Fluids section above when starting fluent from a terminal window command line.

==== On Demand Desktop ====

Compute Node (no GPU requested) or Basic Desktop

::: module load StdEnv/2023 ansys/2025R1
::: runwb2 -oglmesa

Compute Node (with GPU requested)

::: module load StdEnv/2023 ansys/2025R1
::: runwb2

==== Jupyter Lab Desktop ====

Compute Node (no GPU requested)

::: Click to load ansys/2025R1 (or newer version) in the Desktop left hand side menu
::: Click the "Workbench (VNC)" icon located in the Jupyter Lab desktop center window
::: Since the default icon is configured for a gpu node, we must customize it so
::: workbench can be restart in mesa mode.  To proceed, Exit the Workbench desktop,
::: open a terminal window, and run the following commands on the command line:
::: cd ~/Desktop; cp -p $(realpath workbench.desktop) workbench-mesa.desktop
::: then edit workbench-mesa.desktop and change runwb2 -> runwb2 -oglmesa
::: Save the file then click your newly customized icon to start workbench.
::: Note the workbench icon that you created will persist for future sessions
::: until manually deleted with: rm -f ~/Desktop/workbench-mesa.desktop

Compute Node (with GPU requested)

::: Click to load ansys/2025R1 (or newer version) in the Desktop left hand side menu
::: Click the Workbench (VNC) icon located in the Jupyter Lab desktop center window

=== Ensight ===
::: module load StdEnv/2023 ansys/2022R2; A=222; B=5.12.6
::: export LD_LIBRARY_PATH=$EBROOTANSYS/v$A/CEI/apex$A/machines/linux_2.6_64/qt-$B/lib
::: ensight -X

=== Rocky ===
::: module load StdEnv/2023 ansys/2025R1 (or newer versions)
::: Rocky The ansys module handles reading your ~/licenses/ansys.lic
::: RockySolver Run rocky solver directly from command line (add -h for help, untested)
::: RockySchedular Start rocky schedular gui to submit/run jobs on present node (untested)
::: o The SHARCNET license includes Rocky and is therefore free for all researchers to use
::: o Rocky supports GPU-accelerated computing however this capability not been tested or documented yet

== SSH issues ==
::: Some Ansys GUI programs can be run remotely on a cluster compute node by forwarding X over SSH to your local desktop.  Unlike VNC, this approach is untested and unsupported since it relies on a properly setup X display server for your particular operating system OR the selection, installation and configuration of a suitable X client emulator package such as MobaXterm.  Most users will find interactive response times unacceptably slow for basic menu tasks let alone performing more complex tasks such as those involving graphics rendering.  Startup times for GUI programs can also be very slow depending on your Internet connection. For example, in one test it took 40 minutes to fully start the gui up over SSH while starting it with vncviewer required only 34 seconds.  Despite the potential slowness when connecting over SSH to run GUI programs, doing so may still be of interest if your only goal is to open a simulation and perform some basic menu operations or run some calculations. The basic steps are given here as a starting point: 1) ssh -Y username@alliancecan.ca 2) salloc --x11 --time=1:00:00 --mem=16G --cpus-per-task=4 [--gpus-per-node=1] --account=def-mygroup; 3) once connected onto a compute node try running xclock.  If the clock appears on your desktop, proceed to load the desired Ansys module and try running the program.

= Site-specific usage =

== SHARCNET license ==

The SHARCNET Ansys license is free for academic use by any Alliance researcher on any Alliance system.   The installed software does not have any solver or geometry limits.  The SHARCNET license may be used for Publishable Academic Research but not for any private/commercial purposes as this is strictly prohibited by the license terms.  The SHARCNET Ansys license is based on the Multiphysics Campus Solution and includes products such as: HF, EM, Electronics HPC, Mechanical, CFD as described here. ROCKY and LS-DYNA are also now included by the SHARCNET license.  Lumerical acquired by ANSYS in 2020 however is not covered presently although the software is installed with recent ansys modules so can be used with other suitably licensed Ansys servers.  SpaceClaim is not installed Alliance systems (since they are  all linux based) however it is technically covered by the SHARCNET license.  A pool of 1986 anshpc licenses is included with the SHARCNET license to support running large scale parallel simulations with most Ansys products.  To ensure they are used efficiently scaling tests should be run before launching long jobs.  Parallel jobs that do not achieve at least 50% CPU utilization will probably be flagged by the system, resulting in a followup by an Alliance team member.

The SHARCNET Ansys license is made available on a first come first serve basis.  It currently permits each researcher to run a maximum of simultaneous 8 jobs using upto 512 hpc cores.  Therefore any of the following maximum even sized combinations can be run simultaneously 1x512, 2x256, 4x128 or 8x64 across all clusters.  Since the license is oversubscribed there is however the  potential for a shortage of anshpc licenses to develop.  Should a job fail on startup due to a shortage of licenses it will need to be manually be resubmitted.   If over time there are many instances of license shortages reported then either the total job limit per researcher will be decreased (to 6 or 4) and/or the total hpc core limit per researcher will be decreased (to 384 cores or 256) if necessary.  If you need more than 512 hpc cores for your research then consider using the local ANSYS License server at your institution if one is available and contributing towards expanding it if necessary.

Some researchers may prefer to purchase a license subscription from CMC to gain access to their remote license servers to run ansys anywhere besides just on Alliance systems such as in your lab or at home on your laptop.  Doing so will have several benefits 1) a local institutional license server is not needed 2) a physical license does not need to be obtained and reconfigured each year 3) the license can be used almost anywhere including at home, institutions, or any alliance cluster across Canada and 4) installation instructions are provided for Windows machines to enable running spaceclaim (not currently possible on the Alliance clusters since all systems are linux based).  Note however that according to the CMC Ansys Quick Start Guides there may be a 64 core limit per user!

==== License file ====

To use the SHARCNET Ansys license on any Alliance cluster, simply configure your ansys.lic file as follows:

[username@cluster:~] cat ~/.licenses/ansys.lic
setenv("ANSYSLMD_LICENSE_FILE", "1055@license3.sharcnet.ca")
setenv("ANSYSLI_SERVERS", "2325@license3.sharcnet.ca")

==== License query  ====

To show the number of licenses in use by your username and the total in use by all users, run:

ssh graham.computecanada.ca
module load ansys
lmutil lmstat -c $ANSYSLMD_LICENSE_FILE -a | grep "Users of\|$USER"

If you discover any licenses unexpectedly in use by your username (usually due to ansys not exiting cleanly on gra-vdi), connect to the node where it's running, open a terminal window and run the following command to terminate the rogue processes pkill -9 -e -u $USER -f "ansys" after which your licenses should be freed.  Note that gra-vdi consists of two nodes (gra-vdi3 and gra-vdi4) which researchers are randomly placed on when connecting to gra-vdi.computecanada.ca with TigerVNC.  Therefore it's necessary to specify the full hostname (gra-vdi3.sharcnet.ca or grav-vdi4.sharcnet.ca) when connecting with tigervnc to ensure you log into the correct node before running pkill.

=== Local modules ===

When using gra-vdi, researchers have the choice of loading Ansys modules from our global environment (after loading CcEnv) or loading Ansys modules installed locally on the machine itself (after loading SnEnv).  The local modules may be of interest as they include some Ansys programs and versions not yet supported by the standard environment.  When starting programs from local Ansys modules, you can select the CMC license server or continue to use the SHARCNET license server by default.  Settings from ~/.licenses/ansys.lic are only used when dash gui is appended to the ansys program name for instance fluent-gui instead of simply fluent.  Suitable usage of Ansys on gra-vdi : run a single job interactively (in the gui or from the command line) with up to 8 cores and 128G RAM, create or modify simulation input files, post process or visualize data.  To load and use a local ansys module on gra-vdi do the following:

# Connect to gra-vdi.computecanada.ca with TigerVNC.
# Open a new terminal window and load a module:
# module load SnEnv ansys/2024R2 (or older)
# Directly start one of the following ansys programs from the command line:
# runwb2[-gui]|fluent[-gui]|cfx5[-gui]|icemcfd[-gui]|apdl[-gui]|Rocky[-gui]

If you run cfx5-gui your ~/.licenses/ansys.lic file will first be read and then you will get the option to instead select the CMC server and finally choose which CFX program will be started in gui mode from the following :
    1) CFX-Launcher  (cfx5 -> cfx5launch)
    2) CFX-Pre       (cfx5pre)
    3) CFD-Post      (cfdpost -> cfx5post)
    4) CFX-Solver    (cfx5solve)

License feature preferences previously setup with anslic_admin are no longer supported (2021-09-09).  If a license problem occurs, try removing the ~/.ansys directory in your /home account to clear the settings.  If problems persist please contact our technical support and provide the contents of your ~/.licenses/ansys.lic file.

= Additive Manufacturing =

To get started configure your ~/.licenses/ansys.lic file to point to a license server that has a valid Ansys Mechanical License.  This must be done on all systems where you plan to run the software.

== Enable Additive ==

This section describes how to make the Ansys Additive Manufacturing ACT extension available for use in your project. The steps must be performed on each cluster for each ansys module version where the extension will be used. Any extensions needed by your project will also need to be installed on the cluster as described below.  If you get warnings about missing un-needed extensions (such as ANSYSMotion) then uninstall them from your project.

=== Download Extension ===
* download AdditiveWizard.wbex from https://catalog.ansys.com/
* upload AdditiveWizard.wbex to the cluster where it will be used

=== Start Workbench ===
* follow the Workbench section in Graphical use above.
* File -> Open your project file (ending in .wbpj) into Workbench gui

===  Open Extension Manager ===
* click ACT Start Page and the ACT Home page tab will open
* click Manage Extensions and the Extension Manager will open

=== Install Extension ===
* click the box with the large + sign under the search bar
* navigate to select and install your AdditiveWizard.wbex file

=== Load Extension ===
* click to highlight the AdditiveWizard box (loads the AdditiveWizard extension for current session only)
* click lower right corner arrow in the AdditiveWizard box and select Load extension (loads the extension for current AND future sessions)

=== Unload Extension ===
* click to un-highlight the AdditiveWizard box (unloads extension for the current session only)
* click lower right corner arrow in the AdditiveWizard box and select Do not load as default (extension will not load for future sessions)

== Run Additive ==

=== Gra-vdi ===

A user can run a single Ansys Additive Manufacturing job on gra-vdi with up to 16 cores as follows:

* Start Workbench on Gra-vdi as described above in Enable Additive.
* click File -> Open and select test.wbpj then click Open
* click View -> reset workspace if you get a grey screen
* start Mechanical, Clear Generated Data, tick Distributed, specify Cores
* click File -> Save Project -> Solve

Check utilization:
* open another terminal and run: top -u $USER   **OR**  ps u -u $USER | grep ansys
* kill rogue processes from previous runs:  pkill -9 -e -u $USER -f "ansys|mwrpcss|mwfwrapper|ENGINE"

Please note that rogue processes can persistently tie up licenses between gra-vdi login sessions or cause other unusual errors when trying to start gui programs on gra-vdi.  Although rare, rogue processes can occur if an ansys gui session (fluent, workbench, etc) is not cleanly terminated by the user before vncviewer is terminated either manually or unexpectedly - for instance due to a transient network outage or hung filesystem.  If the latter is to blame then the processes may not by killable until normal disk access is restored.

===Cluster===

Project preparation:

Before submitting a newly uploaded Additive project to a cluster queue (with sbatch scriptname) certain preparations must be done.  To begin, open your simulation with Workbench gui (as described in the Enable Additive section above) in the same directory that your job will be submitted from and then save it again. Be sure to use the same ansys module version that will be used for the job.  Next create a Slurm script (as explained in the Cluster Batch Job Submission - WORKBENCH section above).  To perform parametric studies, change Update() to UpdateAllDesignPoints() in the Slurm script.  Determine the optimal number of cores and memory by submitting several short test jobs.  To avoid needing to manually clear the solution and recreate all the design points in Workbench between each test run, either 1) change Save(Overwrite=True) to Save(Overwrite=False) or 2) save a copy of the original YOURPROJECT.wbpj file and corresponding YOURPROJECT_files directory.  Optionally create and then manually run a replay file on the cluster in the respective test case directory between each run, noting that a single replay file can be used in different directories by opening it in a text editor and changing the internal FilePath setting.

 module load ansys/2019R3
 rm -f test_files/.lock
 runwb2 -R myreplay.wbjn

Resource utilization:

Once your additive job has been running for a few minutes, a snapshot of its resource utilization on the compute node(s) can be obtained with the following srun command.  Sample output corresponding to an eight core submission script is shown next.  It can be seen that two nodes were selected by the scheduler:

 [gra-login1:~] srun --jobid=myjobid top -bn1 -u $USER | grep R | grep -v top
   PID USER   PR  NI    VIRT    RES    SHR S  %CPU %MEM    TIME+  COMMAND
 22843 demo   20   0 2272124 256048  72796 R  88.0  0.2  1:06.24  ansys.e
 22849 demo   20   0 2272118 256024  72822 R  99.0  0.2  1:06.37  ansys.e
 22838 demo   20   0 2272362 255086  76644 R  96.0  0.2  1:06.37  ansys.e
   PID USER   PR  NI    VIRT    RES    SHR S  %CPU %MEM    TIME+  COMMAND
  4310 demo   20   0 2740212 271096 101892 R 101.0  0.2  1:06.26  ansys.e
  4311 demo   20   0 2740416 284552  98084 R  98.0  0.2  1:06.55  ansys.e
  4304 demo   20   0 2729516 268824 100388 R 100.0  0.2  1:06.12  ansys.e
  4305 demo   20   0 2729436 263204 100932 R 100.0  0.2  1:06.88  ansys.e
  4306 demo   20   0 2734720 431532  95180 R 100.0  0.3  1:06.57  ansys.e

Scaling tests:

After a job completes, its "Job Wall-clock time" can be obtained from seff myjobid.  Using this value, scaling tests can be performed by submitting short test jobs with an increasing number of cores.  If the Wall-clock time decreases by ~50% when the number of cores is doubled, additional cores may be considered.

= Help resources =

Documentation for recent versions Ansys 202[4|5]R[1|2] is fully available here.  Documentation for older versions such as Ansys 2023R[1|2] however requires login.  Developer documentation can be found in the Ansys Developer Portal. Additional learning resources include the Ansys HowTo videos, the Ansys Educator Educator Hub and the Ansys Webinar series.