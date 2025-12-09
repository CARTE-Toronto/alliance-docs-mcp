---
title: "GNU Parallel/en"
url: "https://docs.alliancecan.ca/wiki/GNU_Parallel/en"
category: "General"
last_modified: "2025-04-03T13:45:32Z"
page_id: 728
display_title: "GNU Parallel"
---

== Introduction ==
GNU Parallel is a tool for running many sequential tasks at the same time on one or more nodes. It is useful for running a large number of sequential tasks, especially if they are short or of variable durations, as well as when doing a parameter sweep. We will only cover the basic options here; for more advanced usage, please see the official documentation.

By default, parallel will run as many tasks as the number of cores allocated by the scheduler, therefore maximizing resource usage. You can change this behaviour using the option --jobs followed by the number of simultaneous tasks that GNU Parallel should run. When one task finishes, a new task will automatically be started by parallel in its stead, always keeping the maximum number of tasks running.

== Basic Usage ==
Parallel uses curly brackets {} as parameters for the command to be run. For example, to run gzip on all the text files in a directory, you can execute
 parallel gzip  }}

An alternative syntax is to use :::, such as in this example:
 ::: $(seq 1 3)
|result=
1
2
3
}}

Note that GNU Parallel refers to each of the commands executed as jobs.  This can be confusing because on many of our systems, a job is a batch script run by a scheduler, and GNU Parallel would be used inside that job.  From that perspective, GNU Parallel's jobs are subjobs.

== Multiple Arguments ==
You can also use multiple arguments by enumerating them, for example:
1 2 ::: $(seq 1 3) ::: $(seq 2 3)
|result=
1 2
1 3
2 2
2 3
3 2
3 3
}}

== File Content as Argument List ==
The syntax :::: takes the content of a file to generate the list of values for the arguments. For example, if you have a list of parameter values in file mylist.txt, you may display its content with:
1 :::: mylist.txt}}

== File Content as Command List ==
GNU Parallel can also interpret the lines of a file as the actual subjobs to be run in parallel, by using redirection. For example, if you have a list of subjobs in file my_commands.txt (one per line), you may run them in parallel as follows:

Note that there are no commands or arguments given to Parallel. This usage mode can be particularly useful if the subjobs contain symbols that are special to GNU Parallel, or the subcommands are to contain a few commands (e.g. cd dir1 && ./executable).

Here is an example how to run a Slurm job using GNU Parallel. The list of commands in my_commands.txt will be run sequentially using 4 CPUs. As soon as one command completes, a new one will be started to maintain the total of 4 commands running at the same time, until the list is exhausted.

==Running on Multiple Nodes==

You can also use GNU Parallel to distribute a workload across multiple nodes in a cluster, such as in the context of a job on our servers. An example of this use is the following:

In this case, we create a file containing the list of nodes, and we use this file to tell GNU Parallel which nodes to use for the distribution of tasks. The --env option allows us to transfer a named environment variable to all the nodes while the --workdir option ensures that the GNU Parallel tasks will start in the same directory as the main node.

For example, when a long list of OpenMP tasks are executed as a single job submitted with --nodes=N, --ntasks-per-node=5 and --cpus-per-task=8, the following command will take into account all processes to be started on all reserved nodes and the number of OpenMP threads per process:
$SLURM_CPUS_PER_TASK
}}

In this case, up to 5*N OpenMP processes are running simultaneously with a CPU usage of up to 800% each.

==Keeping Track of Completed and Failed Commands, and Restart Capabilities==
You can tell GNU Parallel to keep track of which commands have completed by using the --joblog JOBLOGFILE argument. The file JOBLOGFILE will contain the list of completed commands, their start times, durations, hosts, and exit values.  For example,
 parallel --joblog gzip.log gzip  }}

The job log functionality opens the door to a number of possible restart options.  If the parallel command was interrupted (e.g. your job ran longer than the requested walltime for a job), you can make it pick up where it left off using the --resume option, for instance
 parallel --resume --joblog gzip.log gzip  }}
The new jobs will be appended to the old log file.

If some of the subcommands failed (i.e. they produced a non-zero exit code) and you think that you have eliminated the source of the error, you can re-run the failed ones, using --resume-failed, e.g.
 parallel --resume-failed --joblog gzip.log gzip  }}
Note that this will also start subjobs that were not considered before.

==Handling large files==
Let's say we want to count the characters in parallel from a big FASTA file (database.fa) in a task with 8 cores. We will have to use the GNU Parallel --pipepart and --block arguments to efficiently handle chunks of the file.

By varying the block size we get:

 	# Cores in task	Ref. database size	Block read size	# GNU parallel jobs	# Cores used	Time counting chars
1	8              	827MB             	10MB           	83                 	8           	0m2.633s
2	8              	827MB             	100MB          	9                  	8           	0m2.042s
3	8              	827MB             	827MB          	1                  	1           	0m10.877s
4	8              	827MB             	-1             	8                  	8           	0m1.734s

This table shows that choosing the right block size has a real impact on the efficiency and the number of cores actually used.
The first line shows that the block size is too small, resulting in many jobs dispatched over the available cores.
The second line is a better block size, since it results in a number of jobs close to the number of available cores.
The third line shows that the block size is too big and that we are only using 1 core out of 8, therefore inefficiently processing chunks.
Finally, the last line shows that in many cases, letting GNU Parallel adapt and decide on the block size is often faster.

== Running hundreds or thousands of simulations ==
First, you must determine how many resources are required by one simulation, then you can estimate the total resources required in your job.

The submission scripts given in the examples below are based on: 1 serial simulation requiring 2 GB of memory, 1 core and 5 minutes, and 1000 simulations. It would take 83.3 hours or 3.472 days to run with 1 core.

On one node, using 32 cores, it can be completed in 2.6 hours.
One could also use more than one node (see #Running on Multiple Nodes).

=== Arguments List ===
As shown in section #File Content as Argument List, you can use a file containing all the parameters. In this case, parameters are delimited by a tab character (\t) and each line corresponds to one simulation.

=== Commands List ===
As shown in the section #File Content as Command List, you can use a file containing all the commands and their parameters.

=== Multiple Arguments ===
You can use GNU Parallel to generate the parameters and feed them to the command.

==Related topics==
* META
* GLOST
* Job arrays