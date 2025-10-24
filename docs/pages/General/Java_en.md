---
title: "Java/en"
url: "https://docs.alliancecan.ca/wiki/Java/en"
category: "General"
last_modified: "2023-03-31T17:03:42Z"
page_id: 2951
display_title: "Java"
---

`<languages />`{=html}

Java is a general-purpose, high-level, object-oriented programming language developed in 1995 by Sun Microsystems (purchased by Oracle in 2010). One of the principal design goals for Java was a high degree of portability across platforms, summarized by the slogan *write once, run anywhere*, and which is realized by having Java source code compiled to \'bytecode\' which then runs inside a Java virtual machine (JVM), ensuring a very uniform environment across numerous architectures and platforms. This has made Java a popular language choice in some environments and it is also widely used as a language for teaching programming. While performance was not one of the original design goals for Java, there are ways to help Java code run quickly and it has enjoyed a certain popularity in some scientific domains such as the life sciences, e.g. software like the Broad Institute\'s [GATK](https://software.broadinstitute.org/gatk/). This page is not designed to teach the Java programming language but merely to provide some tips and hints for the use of Java in a high-performance computing environment such as Alliance clusters.

Alliance\'s systems have several different Java virtual machines installed which are made available to users via the [ module](https://docs.alliancecan.ca/Utiliser_des_modules/en " module"){.wikilink} command like other software packages. You should normally only have one Java module loaded at a time. The principal commands associated with such Java modules are `java` to launch the Java virtual machine and `javac` to call the Java compiler for converting a Java source file into byte code.

Java software is frequently distributed in the form of a JAR file with the extension `jar`. You can use such software by means of the following command,

assuming the JAR file has been compiled to operate as an autonomous program (i.e. it possesses a `Main-class` manifest header).

## Parallelism in Java {#parallelism_in_java}

### Threading

Java includes built-in support for threading, obviating the need for separate interfaces and libraries like OpenMP, pthreads and Boost threads used in other languages. The principal Java object for handling concurrency is the `Thread` class which a programmer can use by either providing a `Runnable` method to the standard `Thread` class or by subclassing the `Thread` class. As an example of this second approach, consider the following toy program: `{{File
  |name=thread.java
  |lang="Java"
  |contents=
public class HelloWorld extends Thread {
        public void run() {
            System.out.println("Hello World!");
        }
        public static void main(String args[]) {
            (new HelloWorld()).start();
        }
}
}}`{=mediawiki} This second approach is generally the simplest to use but suffers from the drawback that Java does not permit multiple inheritance, so the class which implements multithreading is unable to subclass any other - potentially more useful - class.

### MPI and Java {#mpi_and_java}

One common method for using MPI-style parallelism in a Java program is the [MPJ Express](http://mpj-express.org/) library.

## Pitfalls

### Memory Issues {#memory_issues}

The Java virtual machine expects to have access to all the physical memory on a given compute node, while the scheduler or a shell might impose its own memory limits (often different) according to the submission script specifications or limitations of the login node. In a shared computing environment, these limits ensure that finite resources, such as memory and CPU cores, do not get exhausted by one job at the expense of another.

When the Java VM starts, it sets two memory parameters according to the amount of physical rather than available memory as follows:

- Initial heap size of 1/64 of physical memory
- Maximum heap size of 1/4 of physical memory

On a system with a lot of physical memory, the 1/4 can easily exceed default the memory limits imposed by a shell or by the scheduler, and Java will fail with messages like these

` Could not reserve enough space for object heap`\
` There is insufficient memory for the Java Runtime Environment to continue.`

The two run-time memory parameters, however, can be explicitly controlled on the command line by following either syntax below:

` java -Xms256m -Xmx4g -version`

or

` java -XX:InitialHeapSize=256m -XX:MaxHeapSize=4g -version`

You can see all the command line options the JVM is going to run with by specifying the following flag `-XX:+PrintCommandLineFlags`, like so:

    $ java -Xms256m -Xmx4g -XX:+PrintCommandLineFlags -version
    -XX:InitialHeapSize=268435456 -XX:MaxHeapSize=4294967296 -XX:ParallelGCThreads=4 -XX:+PrintCommandLineFlags -XX:+UseCompressedOops -XX:+UseParallelGC

Alternatively, you can use the `JAVA_TOOL_OPTIONS` environment variable to set the run-time options rather that passing them on the command line. This is especially convenient if you launch multiple Java calls, or call a Java program from another Java program. Here is an example how to do it: \"-Xms256m -Xmx2g\"}} When your Java program is run, it will produce a diagnostic message like this one \"Picked up JAVA_TOOL_OPTIONS\", verifying that the options have been picked up.

Please remember that the Java virtual machine itself creates a memory usage overhead. We recommend specifying the memory limit for your job as 1-2GB more than your setting on the Java command line option -Xmx.

### Garbage Collection {#garbage_collection}

Java uses an automatic system called *garbage collection* to identify variables which are out of scope and return the memory associated with them to the operating system. By default, the Java VM uses a parallel garbage collector (GC) and sets a number of GC threads equal to the number of CPU cores on a given node, whether a Java job is threaded or not. Each GC thread consumes memory. Moreover, the amount of memory each GC thread consumes is proportional to the amount of physical memory. Therefore, we highly recommend matching the number of GC threads to the number of CPU cores you requested from the scheduler in your job submission script, like so `-XX:ParallelGCThreads=12` for example. You can also use the serial garbage collector by specifying the following option `-XX:+UseSerialGC`, whether your job is parallel or not.

### The `volatile` Keyword {#the_volatile_keyword}

This keyword has a sense very different from that which C/C++ programmers are accustomed to. In Java `volatile` when applied to a variable has the effect of ensuring that its value is always read from and written to main memory, which can help to ensure that modifications of this variable are made visible to other threads. That said, there are contexts in which the use of the `volatile` keyword are not sufficient to avoid race conditions and the `synchronized` keyword is required to ensure program consistency.

## Further Reading {#further_reading}

Scott Oaks and Henry Wong, *Java Threads: Understanding and Mastering Concurrent Programming* (3rd edition) (O\'Reilly, 2012)
