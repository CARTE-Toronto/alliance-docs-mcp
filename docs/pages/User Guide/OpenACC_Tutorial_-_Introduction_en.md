---
title: "OpenACC Tutorial - Introduction/en"
url: "https://docs.alliancecan.ca/wiki/OpenACC_Tutorial_-_Introduction/en"
category: "User Guide"
last_modified: "2022-11-30T20:58:07Z"
page_id: 726
display_title: "OpenACC Tutorial - Introduction"
---

`<languages />`{=html}

## CPU vs accelerator {#cpu_vs_accelerator}

Historically, computing has developed around Central Processing Units (CPU) that were optimized for sequential tasks. That is, they would complete only one compute operation during a given clock cycle. The frequency of these units steadily increased until about 2005, when the top speed of the high-end CPUs reached a plateau at around 4 GHz. Since then - for reasons well explained in [this article](https://www.comsol.com/blogs/havent-cpu-clock-speeds-increased-last-years/) - the usual CPU clock frequency has barely moved, and is even now often lower than 4 GHz. Instead, manufacturers started adding multiple computation cores within a single chipset, opening wide the era of parallel computing.

Yet, even as of 2022, sequential tasks still run the fastest on CPUs:

- first, they have direct access to the main computer memory, which can be very large;
- second, because of their very fast clock speed, they can run a small number of tasks very quickly.

But CPUs also have some weaknesses:

- they have relatively low memory bandwidth;
- they use cache mechanisms to mitigate the low bandwidth, but this means that [cache misses](https://en.wikipedia.org/wiki/CPU_cache#Cache_miss) are very costly;
- and they also are rather power-hungry compared to accelerators.

Typical accelerators, such as GPU or coprocessors, are highly parallel chipsets. They are made out of hundreds or thousands of relatively simple and low frequency compute cores. Simply said, they are optimized for parallel computing. High-end GPUs usually have a few thousand compute cores. They also have a high bandwidth to access their own device memory. They present significantly more compute resources than high-end CPUs, and provide a much **higher throughput**, and much **better performance per watt**. However, they embed a relatively low amount of memory, and have a low per-thread performance.

## Porting code to accelerators {#porting_code_to_accelerators}

Porting a code to accelerators can be seen as a phase of an optimization process. A typical optimization process will have the following steps:

1.  Profile the code
2.  Identify bottlenecks
3.  Optimize the most significant bottleneck
4.  Validate the resulting code
5.  Start again from step 1

Much similarly, we can split the task of porting a code to accelerators into the following steps:

1.  Profile the code
2.  Identify parallelism within the bottlenecks
3.  Port the code
    1.  Express parallelism to the compiler
    2.  Express data movement
    3.  Optimize loops
4.  Validate the resulting code
5.  Start again from step 1

OpenACC can be a rather *descriptive* language. This means that the programmer can tell the compiler that he thinks a given portion of the code can be parallelized, and let the compiler figure out exactly how to do it. This is done by adding a few directives to the code (i.e. *express parallelism* in the above list). However, the quality of the compiler will greatly change the achieved performance. Even with the best compilers, there may be unnecessary data movement that needs to get taken out. This is what the programmer will do in the *express data movement* phase. Finally, the programmer may have information that is not available to the compiler which would allow him to achieve better performance by tuning the loops. This is what is done in the *optimize loops* step.

[\^- Back to the lesson plan](https://docs.alliancecan.ca/OpenACC_Tutorial "^- Back to the lesson plan"){.wikilink} \| [Onward to the next unit: *Profiling* -\>](https://docs.alliancecan.ca/OpenACC_Tutorial_-_Profiling "Onward to the next unit: Profiling ->"){.wikilink}
