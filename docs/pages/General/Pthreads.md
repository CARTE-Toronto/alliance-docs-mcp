---
title: "Pthreads/en"
url: "https://docs.alliancecan.ca/wiki/Pthreads/en"
category: "General"
last_modified: "2018-04-27T18:44:02Z"
page_id: 1773
display_title: "Pthreads"
---

=Introduction=
One of the earliest parallelization techniques was through the use of POSIX threads, usually shortened to just pthreads. Like other threading constructs, pthreads parallelization relies on the assumption of a shared memory environment and is, therefore, typically used only on a single node with the number of active threads limited by the number of available CPU cores on that node. While pthreads can
be used with a variety of programming languages, in practice the main target language is C. To parallelize a Fortran program using threads, OpenMP is almost certainly a better idea while C++
programmers would probably find the constructs in the Boost threading library or which are part of the C++11 standard to be more attractive options, given their consistency with object-oriented design.

As one of the earliest forms of parallelization, pthreads have also served as the basis for later approaches to shared memory parallelization like OpenMP and can be thought of as forming a
toolkit of threading primitives that permit the most general and low-level parallelization, at the price of sacrificing much of the simplicity and ease of use of a high level API like OpenMP. The essential
model for pthreads is the dynamic spawning of lightweight sub-processes (threads) that asynchronously carry out operations and then are extinguished by rejoining the program's master process. As all the threads of a program reside in the same memory space, sharing data among them through global variables isn't difficult in comparison with a distributed approach like MPI but any modifications of
this shared data have to be managed with care to avoid  race conditions.

When parallelizing a program using pthreads (or any other technique) it's important to also consider how well the program is able to run in parallel, known as the software's scalability. After you've parallelized your software and are satisfied about its correctness, we recommend that you perform a scaling analysis in order to understand its parallel performance.

=Compilation=
To use the various functions and data structures associated with pthreads in your C program, you will need to include the header file pthread.h and compile your program with a special flag so that it is linked with the pthread library.

There are different ways to specify the number of threads to be used:
* it can be set via a command-line argument;
* it can be set via an environment variable;
* it can be hard-coded into the source file, but then you would not be able to adjust the thread count at run time.

=Creation and Destruction of Pthreads=
When parallelizing an existing serial program using pthreads, we use a programming model where threads are created by a parent, then carry out some work, and finally are reabsorbed or joined back into the parent.  The parent may be the serial master thread or another worker thread.

New threads are created with the function pthread_create.  This function has four arguments:
*the unique identifier for the newly created thread;
*the set of attributes for this thread;
*the C function that the thread will execute upon initiation (the "start routine");
*the argument for the start routine.

This simple example creates twelve threads, each one executing the function task with the argument consisting of the thread's index, from 0 to 11. Note that the call of pthread_create is non-blocking, i.e. the root or master thread, which is executing the main function, continues to execute after each of the twelve worker threads is created. After creating the twelve threads, the master thread then goes into the second for loop and calls pthread_join, a blocking function where the master thread waits for the twelve workers to finish executing the function task and rejoin the master thread. While trivial, this program illustrates the basic lifecycle of a POSIX thread: the master thread creates a thread by assigning it a function to run, then waits for the thread to finish and join back into the execution of the master thread.

If you run this test program several times in a row you'll likely notice that the order in which you see the various worker threads saying hello varies, which is what we would expect since they are now running in an asynchronous manner. Each time the program is executed, the twelve threads compete for access to the standard output during the printf call and from one execution of the program to another the winners of this competition will change.

=Synchronizing Data Access=
In a more realistic program, worker threads will need to read and eventually modify certain data in order to accomplish their tasks. These data normally consist of a set of global variables of different types and dimensions, and with multiple threads reading and writing these data, we need to ensure that the access to these data is synchronized in some fashion to avoid race conditions, i.e. situations in which the program's output depends on the random order in which the asynchronous threads access the data. Typically, we want the parallel version of our program to produce results identical to what we would obtain when running it serially, so the race conditions are unacceptable.

The simplest and most common way to control the reading and writing of data shared among threads is the mutex, derived from the expression 'mutual exclusion'. In pthreads, a mutex is a kind of variable that may be "locked" or "owned" by only one thread at a time. The thread must then release or unlock the mutex once the global data has been read or modified. The code that lies between the call to lock a mutex and the call to unlock it will only be executed by a single thread at a time. To create a mutex in a pthreads program, we declare a global variable of type pthread_mutex_t which must be initialized before it is used by calling pthread_mutex_init. At the program's end we release the resources associated with the mutex by calling pthread_mutex_destroy.

In this example, based on the previous code, access to the standard output channel is serialized - as it normally should be - using a mutex. The call to pthread_mutex_lock is blocking, i.e. the thread will continue to wait indefinitely for the mutex to become available, so you have to take care that no deadlock can occur in your code, that is, that the mutex is guaranteed to become available eventually. This is particularly problematic in a more realistic example where you may have many different mutexes designed to control access to different global data structures. There is also a non-blocking alternative, pthread_mutex_trylock, which if it fails to obtain the mutex lock returns immediately with a non-zero value indicating that the mutex is busy. You should also ensure that no extraneous code appears inside the serialized code block; since this code will be executed in a serial manner, you want it to be as short as it can safely be in order not to reduce your program's parallel performance.

A more subtle form of data synchronization is possible with the read/write lock, pthread_rwlock_t. With this construct, multiple threads can simultaneously read the value of a variable but for write access, the read/write lock behaves like the standard mutex, i.e. no other thread may have have any access (read or write) to the variable. Like with a mutex, a pthread_rwlock_t must be initialized before its first use and destroyed when it is no longer needed during the program. Individual threads can obtain either a read lock by calling pthread_rwlock_rdlock, or a write lock with pthread_rwlock_wrlock. Either one is released using pthread_rwlock_unlock.

Another construct is used to allow multiple threads to wait for a single condition, for example waiting for work to become available for the worker threads. This construct is called a condition variable and has the datatype pthread_cond_t. Like a mutex or read/write lock, a condition variable must be initialized before its first use and destroyed when it is no longer needed. The use of a condition variable also requires a mutex to control access to the variable(s) that are the basis for the condition that is being tested. A thread that needs to wait on a condition will lock the mutex and then call the function pthread_cond_wait with two arguments: the condition variable, and the mutex. The mutex will be released atomically with the creation of the condition variable that the thread is now waiting upon, so that other threads can lock the mutex either to wait on the same condition or to modify one or more variables, thereby changing the condition.

In the above example we have two worker threads which modify the value of the integer workload, whose initial value must be less than or equal to 25. The first thread locks the mutex and then waits because workload <= 25, creating the condition variable ticker and releasing the mutex. The second thread can then perform a loop that increments the value of workload by three at each iteration. After each increment the second thread checks if the workload is greater than 25, and when it is, calls pthread_cond_signal to alert the thread waiting on ticker that the condition is now satisfied.  With the first thread signalled, the second thread sets the exit condition for the loop, releases the mutex, and disappears in the pthread_join. Meanwhile the first thread, having been woken up, increments workload by 15 and exits the function task itself. After the worker threads have been absorbed, the master thread prints out the final value of workload and the program exits. Note that in a more realistic context in which several threads are waiting on a condition variable, we can use pthread_cond_broadcast to notify all the waiting threads that the condition is satisfied. If we use pthread_cond_signal in this context, then a single waiting thread chosen at random will be notified that the condition is satisfied while the others continue to wait.

=Further Reading=
This page is only intended to provide a very brief overview of what is in fact a complex and demanding subject. Individuals who are interested in a more in-depth discussion of pthreads, the various optional arguments that are available for many function calls - where we have used the default NULL argument for such parameters in this page - and advanced topics can consult sources like David Butenhof's Programming with POSIX Threads or the excellent LANL tutorial.