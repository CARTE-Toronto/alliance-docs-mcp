---
title: "Scalability/en"
url: "https://docs.alliancecan.ca/wiki/Scalability/en"
category: "General"
last_modified: "2019-02-12T16:03:04Z"
page_id: 7232
display_title: "Scalability"
---

`<languages />`{=html} In the context of parallel programming, [scalability](https://en.wikipedia.org/wiki/Scalability) refers to the capacity of a program to efficiently use added computing resources, i.e. CPU cores. One might naively imagine that doubling the number of CPU cores devoted to a calculation will halve the duration of the calculation, this is rarely the case. Instead we observe that the gain in performance depends on the nature of the problem, the algorithm or program used to solve it, the underlying hardware (notably memory and network), and the number of CPU cores being used. For this reason when you are planning to use a parallel program on a particular cluster we recommend that you conduct a scalability analysis where the software is tested using a fixed problem while varying the number of CPU cores according to some method (e.g. 2, 4, 8, 16, 32, 64 cores). The run time is obtained for each number of cores, and the resulting curve plotted.

Why is the scalability usually worse than what we might hope for? There are two major reasons:

Firstly, in the parallelization of the code not every operation can be done in parallel and so some percentage of the program\'s execution remains serial. This percentage represents an ultimate limit for the parallel efficiency of the software. Suppose the serial version of some program needs an hour to do a calculation, and six minutes of that (10%) are spent in operations which cannot be parallelized. Even with an infinite number of cores we could not have the program\'s duration sink below six minutes. The best we can hope for is that this \"serial percentage\" shrinks as we increase the size of the problem the software is working on.

Secondly, the parallelization of the program normally requires a certain amount of communication and synchronization among the parallel processes and the cost of this \"parallel overhead\" will increase with the number of processes working together, typically as a power of the number of cores, $T_c \propto n^\alpha$ where $\alpha > 1$. If we now suppose that the scientific part of the program\'s run time is divided equally among the number of cores apart from a residual serial part, so $T_s = A + B/n$, the total duration of the program $T = T_s + T_c = A + B/n + C n^\alpha$ (with $A$, $B$ and $C$ positive real numbers whose value depends on the particular cluster, program and test problem) will ultimately be dominated by this final parallel overhead factor as $n\to\infty$. In the case where $A$ and $B$ are much larger than $C$, when we plot the curve of the run time versus the number of CPU cores we will obtain something that looks like the accompanying figure. ![](Scaling_plot.png "Scaling_plot.png") The most important point to note about this curve is that while for smaller numbers of cores the run time falls, at a certain number of cores a minimum is reached (for $n\approx 22$), and after that the program duration starts to *increase* as we add more processes: too many cooks spoil the broth, according to the proverb. When you are using a parallel program, it\'s crucial to carry out such a scalability analysis in order to know, for the nature and size of problem you\'re working on and the cluster you\'re using, what is the optimal choice of the number of CPU cores: 4, 128, 1024, or some other figure?

It is up to you to choose a test problem for the scalability analysis. You want a problem that is relatively small so that these tests can be carried out quickly, but not so small as to be completely unrepresentative of a production run. A test problem that requires 30 to 60 minutes to finish on one or two cores is probably a good choice. One which runs in under ten minutes is almost certainly too short to be of value. In certain contexts, such as an analysis of the program\'s behaviour under weak scaling (see below), you also want to have a test problem whose size can be easily increased, ideally in a fairly continuous manner.

There is one class of problems for which the factor $C$ is for all practical purposes zero, so that there is no parallel overhead to speak of. Such problems are called \"embarrassingly parallel\". A good example might be running an analysis on 500 different files, in which the analysis of an individual file is independent of any others and simply generates a single number that can be stored in an array. In this case there is no need to synchronize the operations of the various processes analyzing the files nor will any communication among these processes be necessary, so that we can achieve perfect scaling out to any number of processes; the only limitation is the number of files we have.

In the next two sections, we will consider two different forms of scaling, strong and weak. When the term scaling is used without any qualification \"strong scaling\" is normally what is meant. However, weak scaling may be more important depending on how you intend to use the multiple cores.

- Do you wish to perform the same size of simulations as before, but more quickly? Then strong scaling applies.
- Or do you wish to simulate larger or more detailed models, and are willing to wait just as long as before, but for better results? Then weak scaling applies.

## Strong scaling {#strong_scaling}

In this case the problem to be used for the scalability analysis is fixed while the number of CPU cores increases. Ideally, we would expect the scaling to be linear, i.e. the decrease in the program\'s run time compared to some reference value is the reciprocal of the number of cores added compared to that used for the reference calculation. As a concrete example of doing an analysis of the strong scalability of a program, imagine a parallel program which we have tested on the same cluster using the same input parameters, obtaining the results in the table below:

  \|Cores   Run Time (s)   Efficiency (%)
  --------- -------------- ----------------
  2         2765           N/A
  4         1244           111.1
  8         786            87.9
  16        451            76.6
  32        244            70.8
  64        197            44.0
  128       238            18.2

The efficiency in this table is calculated by dividing the reference run time at two cores by the run time at $n$ cores, then dividing the result by $n/2$ and finally multiplying by a hundred to get a percentage. This percentage measures the degree to which the parallel performance scales linearly, i.e. doubling the number of cores halves the run time, which corresponds to an efficiency of 100%.

In the table above, we notice that when going from 2 to 4 cores, we achieve greater than 100% efficiency. This is called \"superlinear scaling\". It occurs rarely, but when it does it is usually due to the presence of a CPU cache which functions more effectively as each CPU core has less to do.[^1]

The test with 128 cores actually took longer than with 64 cores, 238 seconds versus 197 seconds. The 128-core efficiency is therefore terrible, only 18%.

An efficiency of 75% or more is good, so we would advise the user of this software with input like this test case to submit jobs which use 16 CPU cores. The run time does continue to decrease up to 64 cores, but the improvement in run time beyond 16 cores would not be a good use of resources.

The number and range of data points that you obtain for your scalability analysis is up to you. We recommend at least five or six values, although if you find the program runs more slowly with added cores, you should obviously not pursue the analysis beyond that number of cores.

## Weak scaling {#weak_scaling}

In *weak scaling*, the problem size is increased in proportion to the increase in the number of CPU cores so that in an ideal situation of linear scaling the program\'s run time will always remain the same. The definition of \"problem size\" depends on the nature of the problem: in a molecular simulation it might be the number of atoms, in a fluid dynamics simulation it might be the number of cells or nodes in the mesh. We can create a table of run times as in the preceding section, increasing the problem size by an amount equal to the increase in the number of cores:

  \|Cores   Problem Size   Run Time (s)   Efficiency (%)
  --------- -------------- -------------- ----------------
  1         1000           3076           \-
  4         4000           3078           99.9
  12        12,000         3107           99.0
  48        48,000         3287           93.6
  128       128,000        3966           77.6

The formula for the efficiency here is quite simple, just the reference run time divided by the run time at $n$ cores then multiplied by a hundred to obtain a percentage. Once again, the goal is to achieve an efficiency of at least 75%. As is often the case, efficiency remains high up to larger core counts than with strong scaling.

Weak scaling tends to be especially pertinent for applications that are memory-bound. If the parallel program has been designed to privilege communications between nearest neighbours then the weak scaling is usually good. An application which performs a lot of nonlocal communication (e.g. a fast Fourier transform[^2]) may exhibit poor performance in a weak scalability analysis.

[^1]: Wikipedia, \"Super-linear speedup\": <https://en.wikipedia.org/wiki/Speedup#Super-linear_speedup>

[^2]: Wikipedia, \"Fast Fourier transform: <https://en.wikipedia.org/wiki/Fast_Fourier_transform>
