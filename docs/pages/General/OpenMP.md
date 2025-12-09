---
title: "OpenMP/en"
url: "https://docs.alliancecan.ca/wiki/OpenMP/en"
category: "General"
last_modified: "2024-07-15T19:47:44Z"
page_id: 1831
display_title: "OpenMP"
---

== Description ==
OpenMP (Open Multi-Processing) is an application programming interface (API) for shared memory parallel computing. It is supported on numerous platforms, including Linux and Windows, and is available for the C/C++ and Fortran programming languages. The API consists of a set of directives, a software library, and environment variables.

OpenMP allows one to develop fine-grained parallel applications on a multicore machine while making it possible to preserve the structure of the serial code. Although there is only one program instance running, it can execute multiple subtasks in parallel. Directives inserted into the program control whether a section of the program executes in parallel, and if so, they also control the distribution of work among subtasks. The beauty of these directives is that they are usually non-intrusive. A compiler that does not support them can still compile the program and the user can run it serially.

OpenMP relies on the notion of threads. A thread is a bit like a light weight process or a "virtual processor, operating serially", and can formally be defined as the smallest unit of work/processing that can be scheduled by an operating system. From a programmer's point of view, if there are five threads, then that corresponds virtually to five cores that can do a computation in parallel. It is important to understand that the number of threads is independent of the number of physical cores within the computer. Two cores can, for example, run a program with ten threads. The operating system decides how to share the cores' time between threads.

Conversely, one thread can not be executed by two processors at the same time, so if you have (e.g.) four cores available you must create at least four threads in order to take advantage of them all. In some cases it may be advantageous to use more threads than the number of available cores, but the usual practice is to match the number of threads to the number of cores.

Another important point concerning threads is synchronization. When multiple threads in a program do computations at the same time, one must assume nothing about the order in which things happen. If the order matters for the correctness of the code, then the programmer must use OpenMP synchronization directives to achieve that. Also, the precise distribution of threads over cores is unknown to the programmer (although thread affinity capabilities are available to control that).

When parallelizing a program using OpenMP (or any other technique) it's important to also consider how well the program is able to run in parallel, known as the software's scalability. After you've parallelized your software and are satisfied about its correctness, we recommend that you perform a scaling analysis in order to understand its parallel performance.

The following link points to a tutorial for getting started with OpenMP under Linux.

== Compilation ==
For most compilers, compiling an OpenMP program is done by simply adding a command-line option to the compilation flags. For the GNU compilers (GCC) it is -fopenmp, but for Intel depending on the version it may be -qopenmp, -fopenmp, or -openmp.  Please refer to your specific compiler's documentation.

== Directives ==

OpenMP directives are inserted in Fortran programs using sentinels. A sentinel is a keyword placed immediately after a symbol that marks a comment. For example:

!$OMP directive
c$OMP directive
C$OMP directive
*$OMP directive

In C, directives are inserted using a pragma construct, as follows:

#pragma omp directive

===OpenMP directives===

Fortran                                   	C, C++
!$OMP PARALLEL [clause, clause,…]
block  	$OMP END PARALLEL                 	#pragma omp parallel [clause, clause,…]
structured-block
!$OMP DO [ clause, clause,… ]
do_loop     	$OMP END DO                       	#pragma omp for [ clause, clause,… ]
for-loop
!$OMP SECTIONS [clause, clause,…]         	$OMP SECTION
block                	$OMP SECTION
block	$OMP END SECTIONS [NOWAIT]	#pragma omp sections [clause, clause,…] {
[ #pragma omp section ]
structured-block
[ #pragma omp section ]
structured-block
}
!$OMP SINGLE [clause, clause,…]
block     	$OMP END SINGLE [NOWAIT]          	#pragma omp single [clause, clause,…]
structured-block
!$OMP PARALLEL DO [clause, clause,…]
DO_LOOP
[ !$OMP END PARALLEL DO ]	#pragma omp parallel for [clause, clause,…]
for-loop
!$OMP PARALLEL SECTIONS [clause, clause,…]	$OMP SECTION
block                	$OMP SECTION
block	$OMP END PARALLEL SECTIONS	#pragma omp parallel sections [clause, clause,…] {
[ #pragma omp section ]
structured-block
[ #pragma omp section ]
structured-block
}
!$OMP MASTER
block                        	$OMP END MASTER                   	#pragma omp master
structured-block
!$OMP CRITICAL [(name)]
block             	$OMP END CRITICAL [(name)]        	#pragma omp critical [(name)]
structured-block
!$OMP BARRIER                             	#pragma omp barrier
!$OMP ATOMIC
expresion_statement          	#pragma omp atomic
expression-statement
!$OMP FLUSH [(list)]                      	#pragma omp flush [(list)]
!$OMP ORDERED
block                       	$OMP END ORDERED                  	#pragma omp ordered
structured-block
!$OMP THREADPRIVATE( /cb/[, /cb/]…)       	#pragma omp threadprivate ( list )
Clauses                                   	Clauses
PRIVATE ( list )                          	private ( list )
SHARED ( list )                           	shared ( list )
SHARED | NONE )                           	none )
FIRSTPRIVATE ( list )                     	firstprivate ( list )
LASTPRIVATE ( list )                      	lastprivate ( list )
intrinsic } : list )                      	reduction ( op : list )
IF ( scalar_logical_expression )          	if ( scalar-expression )
COPYIN ( list )                           	copyin ( list )
NOWAIT                                    	nowait

== Environment ==
There are a few environment variables that influence the execution of an OpenMP program:

OMP_NUM_THREADS
OMP_SCHEDULE
OMP_DYNAMIC
OMP_STACKSIZE
OMP_NESTED

They can be set and modified using a UNIX command such as
12}}

In most cases, you want to set OMP_NUM_THREADS to the number of reserved cores per machine though this could be different for a hybrid OpenMP/MPI application.

The second most important environment variable is probably OMP_SCHEDULE. This one controls how loops (and, more generally, parallel sections) are distributed. The default value depends on the compiler, and can also be added into the source code. Possible values are static,n, dynamic,n, guided,n or auto. For the first three cases, n corresponds to the number of iterations managed by each thread. For the static case, the number of iterations is fixed, and iterations are distributed at the beginning of the parallel section. For the dynamic case, the number of iterations is fixed, but they are distributed during execution, as a function of the time required by each thread to execute its iterations. For the guided case, n corresponds to the minimal number of iterations. The number of iterations is first chosen to be "large", but dynamically shrinks gradually as the remaining number of iterations diminishes. For the auto mode, the compiler and the library are free to choose what to do.

The advantage of the cases dynamic, guided and auto, is that they theoretically allow a better load-balancing of the threads as they dynamically adjust the work assigned to each thread. Their disadvantage is that the programmer does not know in advance on which core a certain thread executes, and which memory it will need to access. Hence, with this kind of scheduling, it is impossible to predict the affinity between memory and the executing core. This can be particularly problematic in a
NUMA architecture.

The OMP_STACKSIZE environment variable specifies the size of the stack for each thread created by the OpenMP runtime. Note that the main OpenMP thread (executing the sequential part of the OpenMP program) gets its stack size from the execution shell, while OMP_STACKSIZE applies to each additional thread created at runtime. If OMP_STACKSIZE is not set, its implied value will be 4M. If your OpenMP code does not have enough stack memory, it might crash with a segmentation fault error message.

Other environment variables are also available. Certain variables are specific to a compiler whereas others are more generic. For an exhaustive list for Intel compilers, please see the following web site, and for GNU compilers, see this one.

Environment variables specific to the Intel compiler start with KMP_ whereas those specific to Gnu start with GOMP_. For optimal performance regarding memory access, it is important to set the
OMP_PROC_BIND variable as well as the affinity variables, KMP_AFFINITY for Intel, and GOMP_CPU_AFFINITY for GNU compilers. This prevents the movement of OpenMP threads between processors by the operating system. This is particularly important in a NUMA architecture found in most modern computers.

== Example ==
Here is a Hello world example that shows the use of OpenMP.

Compiling and running the C code goes as follows:
 litai10:~$ gcc -O3 -fopenmp ompHello.c -o ompHello
 litai10:~$ export OMP_NUM_THREADS=4
 litai10:~$ ./ompHello
 Hello world from thread 0 out of 4
 Hello world from thread 2 out of 4
 Hello world from thread 1 out of 4
 Hello world from thread 3 out of 4

Compiling and running the Fortran 90 code is as follows:
 litai10:~$ gfortran -O3 -fopenmp ompHello.f90 -o fomphello
 litai10:~$ export OMP_NUM_THREADS=4
 litai10:~$ ./fomphello
 Hello world from thread           0 out of           4
 Hello world from thread           2 out of           4
 Hello world from thread           1 out of           4
 Hello world from thread           3 out of           4

For an example of how to submit an OpenMP job, see Running jobs.

== References ==
Lawrence Livermore National Labs has a comprehensive tutorial on OpenMP.

OpenMP.org publishes the formal specifications,
handy reference cards for the C/C++ and Fortran interfaces, and examples.