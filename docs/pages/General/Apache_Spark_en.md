---
title: "Apache Spark/en"
url: "https://docs.alliancecan.ca/wiki/Apache_Spark/en"
category: "General"
last_modified: "2024-08-29T23:04:12Z"
page_id: 9508
display_title: "Apache Spark"
---

`<languages />`{=html}

# Introduction

Apache Spark is an open source framework for distributed computation initially developed by the AMPLab at Berkeley University and is now a project sponsored by the Apache foundation. Unlike the MapReduce algorithm implemented by Hadoop that uses disk storage, Spark makes use of primitives which are stored in memory, thereby achieving up to 100x the performance of Hadoop in certain applications. Loading data in memory allows them to be queried frequently, making Spark a framework especially appropriate for automated learning and interactive data analysis.

# Usage

## PySpark

```{=mediawiki}
{{File
  |name=pyspark_submit.sh
  |lang="sh"
  |lines=yes
  |contents=
#!/bin/bash
#SBATCH --account=def-someuser
#SBATCH --time=00:01:00
#SBATCH --nodes=4
#SBATCH --mem=4G
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1

module load spark/2.3.0
module load python/3.7

# Recommended settings for calling Intel MKL routines from multi-threaded applications
# https://software.intel.com/en-us/articles/recommended-settings-for-calling-intel-mkl-routines-from-multi-threaded-applications 
export MKL_NUM_THREADS=1
export SPARK_IDENT_STRING=$SLURM_JOBID
export SPARK_WORKER_DIR=$SLURM_TMPDIR
export SLURM_SPARK_MEM=$(printf "%.0f" $((${SLURM_MEM_PER_NODE} *95/100)))

start-master.sh
sleep 5
MASTER_URL=$(grep -Po '(?=spark://).*' $SPARK_LOG_DIR/spark-${SPARK_IDENT_STRING}-org.apache.spark.deploy.master*.out)

NWORKERS=$((SLURM_NTASKS_PER_NODE * SLURM_JOB_NUM_NODES - 1))
SPARK_NO_DAEMONIZE=1 srun -n ${NWORKERS} -N ${NWORKERS} --label --output=$SPARK_LOG_DIR/spark-%j-workers.out start-slave.sh -m ${SLURM_SPARK_MEM}M -c ${SLURM_CPUS_PER_TASK} ${MASTER_URL} &
slaves_pid=$!

srun -n 1 -N 1 spark-submit --master ${MASTER_URL} --executor-memory ${SLURM_SPARK_MEM}M $SPARK_HOME/examples/src/main/python/pi.py

kill $slaves_pid
stop-master.sh
}}
```
## Java Jars {#java_jars}

```{=mediawiki}
{{File
  |name=pyspark_java_submit.sh
  |lang="sh"
  |lines=yes
  |contents=
#!/bin/bash
#SBATCH --account=def-someuser
#SBATCH --time=00:01:00
#SBATCH --nodes=4
#SBATCH --mem=4G
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1

module load spark/2.3.0

# Recommended settings for calling Intel MKL routines from multi-threaded applications
# https://software.intel.com/en-us/articles/recommended-settings-for-calling-intel-mkl-routines-from-multi-threaded-applications 
export MKL_NUM_THREADS=1
export SPARK_IDENT_STRING=$SLURM_JOBID
export SPARK_WORKER_DIR=$SLURM_TMPDIR
export SLURM_SPARK_MEM=$(printf "%.0f" $((${SLURM_MEM_PER_NODE} *95/100)))

start-master.sh
sleep 5
MASTER_URL=$(grep -Po '(?=spark://).*' $SPARK_LOG_DIR/spark-${SPARK_IDENT_STRING}-org.apache.spark.deploy.master*.out)

NWORKERS=$((SLURM_NTASKS_PER_NODE * SLURM_JOB_NUM_NODES - 1))
SPARK_NO_DAEMONIZE=1 srun -n ${NWORKERS} -N ${NWORKERS} --label --output=$SPARK_LOG_DIR/spark-%j-workers.out start-slave.sh -m ${SLURM_SPARK_MEM}M -c ${SLURM_CPUS_PER_TASK} ${MASTER_URL} &
slaves_pid=$!

SLURM_SPARK_SUBMIT="srun -n 1 -N 1 spark-submit --master ${MASTER_URL} --executor-memory ${SLURM_SPARK_MEM}M"
$SLURM_SPARK_SUBMIT --class org.apache.spark.examples.SparkPi $SPARK_HOME/examples/jars/spark-examples_2.11-2.3.0.jar 1000
$SLURM_SPARK_SUBMIT --class org.apache.spark.examples.SparkLR $SPARK_HOME/examples/jars/spark-examples_2.11-2.3.0.jar 1000

kill $slaves_pid
stop-master.sh
}}
```
# Monitoring

The event log for the Spark application which was executed can be saved for later examination by means of a Web application provided by Spark. The following instructions show how to activate the saving of the event logs and how to start the Web application.

## Configuration

First create a directory which will contain the event log of the application:

If it doesn\'t already exist, next create a directory which will contain the Spark configuration parameters:

In this directory, create the following file or add the content shown to the file `spark-defaults.conf` if it already exists.

## Visualization

Create a [tunnel](https://docs.alliancecan.ca/SSH_tunnelling "tunnel"){.wikilink} between your computer and the cluster.

Load the Spark module:

Start the Web application for visualizing the event log: 1 start-history-server.sh \|result= starting org.apache.spark.deploy.history.HistoryServer, logging to /home/`<userid>`{=html}/.spark/`<spark version>`{=html}/log/spark-`<userid>`{=html}-org.apache.spark.deploy.history.HistoryServer-1-`<server>`{=html}.computecanada.ca.out Spark Command: /cvmfs/soft.computecanada.ca/easybuild/software/2017/Core/java/1.8.0_121/bin/java -cp /home/`<userid>`{=html}/.spark/`<spark version>`{=html}/conf/:/cvmfs/soft.computecanada.ca/easybuild/software/2017/Core/spark/2.2.0/jars/\* -Xmx1g org.apache.spark.deploy.history.HistoryServer ======================================== Using Spark\'s default log4j profile: org/apache/spark/log4j-defaults.properties 17/10/13 04:28:56 INFO HistoryServer: Started daemon with process name: 71616@`<server>`{=html}.computecanada.ca 17/10/13 04:28:56 INFO SignalUtils: Registered signal handler for TERM 17/10/13 04:28:56 INFO SignalUtils: Registered signal handler for HUP 17/10/13 04:28:56 INFO SignalUtils: Registered signal handler for INT 17/10/13 04:28:56 INFO SecurityManager: Changing view acls to: `<userid>`{=html} 17/10/13 04:28:56 INFO SecurityManager: Changing modify acls to: `<userid>`{=html} 17/10/13 04:28:56 INFO SecurityManager: Changing view acls groups to: 17/10/13 04:28:56 INFO SecurityManager: Changing modify acls groups to: 17/10/13 04:28:56 INFO SecurityManager: SecurityManager: authentication disabled; ui acls disabled; users with view permissions: Set(`<userid>`{=html}); groups with view permissions: Set(); users with modify permissions: Set(`<userid>`{=html}); groups with modify permissions: Set() 17/10/13 04:28:56 INFO FsHistoryProvider: History server ui acls disabled; users with admin permissions: ; groups with admin permissions 17/10/13 04:29:01 WARN NativeCodeLoader: Unable to load native-hadoop library for your platform\... using builtin-java classes where applicable 17/10/13 04:29:02 INFO FsHistoryProvider: Replaying log path: <file:/home/>`<userid>`{=html}/.spark/`<spark version>`{=html}/eventlog/app-20171013040359-0000 17/10/13 04:29:02 INFO Utils: Successfully started service on port 18080. 17/10/13 04:29:02 INFO HistoryServer: Bound HistoryServer to 0.0.0.0, and started at <http://>`<server ip address>`{=html}:18080

}} Copy the URL which is shown in the terminal and paste it in your Web browser.

To stop the visualization application, type the combination Ctrl-C in the terminal used to launch the application.
