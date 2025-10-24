---
title: "GDB/en"
url: "https://docs.alliancecan.ca/wiki/GDB/en"
category: "General"
last_modified: "2024-07-12T14:54:51Z"
page_id: 9401
display_title: "GDB"
---

`<languages />`{=html} [GDB](https://www.gnu.org/software/gdb/) is a debugger used to investigate software problem. GDB is an acronym saying `<i>`{=html}GDB: The `<u>`{=html}G`</u>`{=html}NU Project `<u>`{=html}D`</u>`{=html}e`<u>`{=html}b`</u>`{=html}ugger`</i>`{=html}

## Description

With a debugger, it is possible to quickly find the cause of a problem in a piece of software. Often it is used to resolve [segmentation faults](https://en.wikipedia.org/wiki/Segmentation_fault). If you desire to resolve a problem relating to memory (for example, a [memory leak](https://en.wikipedia.org/wiki/Memory_leak)), we recommend using [ Valgrind](https://docs.alliancecan.ca/Debugging_and_profiling#Valgrind " Valgrind"){.wikilink}.

## Use cases {#use_cases}

### Finding a bug with the debugger {#finding_a_bug_with_the_debugger}

In this section, the following program is used: `{{File
|name= program.cpp 
|lines=yes
|contents=
#include <iostream>
#include <vector>
using namespace std;

int main(int argc, char**argv) {
    
    vector<int> numbers;

    int iterator = 1000;

    while(iterator --) {
        numbers.push_back(iterator);
    }

    cout << numbers[1000000] << endl;

    return 0;
}
}}`{=mediawiki} This program generates a segmentation fault when it is ran.

We may then run the program inside the debugger. Note that we compiled using the option `-g` to include debugging symbols within the binary and allow the debugger to provide more information on the bug. We run the program inside the debugger using 2, argv0x7fffffffda88) at program.cpp:15 15 cout \<\< numbers\[1000000\] \<\< endl; Missing separate debuginfos, use: debuginfo-install glibc-2.16-31.fc18.x86_64 libgcc-4.7.2-8.fc18.x86_64 libstdc++-4.7.2-8.fc18.x86_64

(gdb) bt

1.  0 0x0000000000400c17 in main (argc2, argv0x7fffffffda88) at program.cpp:15

\| }} So, the above error is caused by line 15. The code tries to use index 1000000, but the array only contains 1000 elements.

### Finding the cause of a segmentation fault using a `core` file {#finding_the_cause_of_a_segmentation_fault_using_a_core_file}

In this example, we use the same program as in the previous section. We however do so without using the debugger directly. This is useful for a bug that happens a long time after the program has started.

To find the cause for this error, a `core` file must be generated. To do this, you must activate the creation of such files.

Executing the same program again, a `core` file is written.

Using the `program` binary executable and the `core` file, it is possible to trace its execution up to the error. 1, argv0x7fff2315c848) at program.cpp:15 15 cout \<\< numbers\[1000000\] \<\< endl; Missing separate debuginfos, use: debuginfo-install glibc-2.16-31.fc18.x86_64 libgcc-4.7.2-8.fc18.x86_64 libstdc++-4.7.2-8.fc18.x86_64

(gdb) bt

1.  0 0x0000000000400c17 in main (argc1, argv0x7fff2315c848) at

program.cpp:15 \| }} We here get the same result as if we had run it inside the debugger.

### Attaching the debugger to a running process {#attaching_the_debugger_to_a_running_process}

It is possible to debug a process that is already running, for example a job running on one of the compute nodes. To do so, we first need the process ID.

`grep firefox  grep -v grep`

seb 12691 6.4 7.5 1539672 282656 ? Sl 08:53 6:48 /usr/lib64/firefox/firefox <http://www.google.ca/> }}

After that, it is possible to attach the debugger directly.

After having done this, a lot of information is displayed.

There are many commands available within GDB. One of the most useful is `backtrace`, or `bt`. This commands shows the current call stack.

## Advanced usage {#advanced_usage}

In the previous sections, we used the `run` and `backtrace` commands. Many more commands are available to debug in an interactive way, by stopping the program. For example, you can set breakpoints on functions or lines of code, or whenever a given variable is modified. When execution is interrupted, you can analyse the state of the program by printing the value of variables. The following table contains a list of the main commands.

  \| Command          \| Shortcut   \| Argument                     \| Description
  ------------------- ------------- ------------------------------- ----------------------------------------------------------
  run/kill            r/k           \-                              begin/stop execution
  where / backtrace   bt            \-                              displays the backtrace
  break               b             src.c:line_number or function   sets a break point at the given line of code or function
  watch               \-            variable name                   interrupts the program when a variable is modified
  continue            c             \-                              resume the program
  step                s             \-                              execute the next operation
  print               p             variable name                   displays the content of a variable
  list                l             src.c:number                    displays the given line of code

  : align=\"bottom\" style=\"color:#e76700;\"\|*Main GDB commands*

### Displaying STL structures {#displaying_stl_structures}

By default, GDB does not display C++ STL structures very well. Many solutions are given [here](https://sourceware.org/gdb/wiki/STLSupport). The simplest solution is probably [this one](http://www.yolinux.com/TUTORIALS/GDB-Commands.html#STLDEREF), which is to copy [this file](http://www.yolinux.com/TUTORIALS/src/dbinit_stl_views-1.03.txt) in your home folder, with the name `~/.gdbinit`.

## Other resources {#other_resources}

- [GDB website](http://www.sourceware.org/gdb/)
- [GDB tutorial](http://oucsace.cs.ohiou.edu/~bhumphre/gdb.html)
- [Talk from the TACC on debugging and profiling](http://goo.gl/rLPvR0)
