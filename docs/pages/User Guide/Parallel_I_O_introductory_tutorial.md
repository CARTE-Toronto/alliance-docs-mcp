---
title: "Parallel I/O introductory tutorial/en"
url: "https://docs.alliancecan.ca/wiki/Parallel_I/O_introductory_tutorial/en"
category: "User Guide"
last_modified: "2025-08-20T15:23:03Z"
page_id: 25786
display_title: "Parallel I/O introductory tutorial"
---

This tutorial will discuss issues in handling large amounts of data in HPC, and a variety of parallel I/O strategies for doing large-scale input/output (I/O) with parallel jobs. In particular we will describe the use of MPI-IO, and of parallel I/O libraries such as NetCDF and HDF5.

=Issues & goal=

Many of today’s problems require large parallel runs on large distributed-memory machines (clusters). Broadly there are three important I/O activities in these calculations.  
# The application must read the initial dataset or conditions from one or more files.
# The application state may need to be written to a file for restarting the application in case of some kind of failure.  This is called checkpointing.
# Results need to be stored for follow-up runs or post-processing.

The figure below shows a simple sketch of the I/O bottleneck problem when using many CPUs in a parallel job.  Amdahl’s law says the speedup of a parallel program is limited by the time needed for the sequential fraction of the program.  So if the I/O part in the application works sequentially as shown, the performance of the code will be not as great as desired.

Efficient I/O without stressing out the storage system--- even a high-performance storage system--- is challenging.

* Total Execution Time  = Computation Time + Communication Time + I/O time
* Optimize all the components of the equation above to get best performance!
* Individual load and store operations are more time-consuming than individual arithmetic operations.
* In some cases, total execution time is dominated by I/O time.  That is our focus in this article.

==Disk access rates over time==

In HPC systems, I/O-related systems are often slow as compared to other parts. The figure below shows how the relative speed of two important components changed over several decades.  From 1956 to 2014 the speed of storage (the purple line) increased by a little over 4 orders of magnitude.  In less than half that time (1993 to 2014) the speed of the top supercomputers in the world (the green line) increased by over five orders of magnitude.  This discrepancy explains can produce data at a rate much, much faster than we can store it.  Therefore we need to pay special attention to how to store the data appropriately.

==How to calculate I/O speed==

Before we proceed more, we better make sure two following performance measurements. Firstly, there is ‘IOPs’. IOPs means I/O operations per second. The operation includes read/write and so on and IOPs is an inverse of latency (think about period (latency) and frequency(IOPs)). And also there is ‘I/O Bandwidth’. The bandwidth is defined as ‘quantity you read/write’. I believe all of you are quite used to this terminology from Internet connection at your home or office. Anyway, here is an information chart for several I/O devices. As you can see, Top-of-the-line SSDs on a PCI Express can push to unto 1GB IOPs. However, the device is still very expensive so it’s not a right fit for several hundreds terabyte supercomputing systems.

One thing I would like to emphasize is that parallel filesystems are optimized for efficient I/O by multiple users on multiple machines/nodes. So, it does not result in “supercomputing” performance in I/O.

*IOPs = Input / Output operations per second (read/write/open/close/seek) ; essentially an inverse of latency
*I/O Bandwidth = quantity you read / write

Parallel (distributed) filesystems are optimized for efficient I/O by multiple users on multiple machines/nodes, do not result in “supercomputing” performance

* disk-access time + communication over the network  (limited bandwidth, many users)

==I/O Software + Hardware stack==

* I/O Hardware --> Parallel filesystem --> I/O Middleware --> High-end I/O library --> Application

When it comes to organizing parallel I/O, there are several layers of abstraction you should keep in mind. First of all, let’s start from the bottom. There is a I/O hardware which is a physical array or hard-disks attached to the cluster. On top of that, we are running parallel file system.

On most of the national systems we are running Lustre which is an open-source filesystem. The purpose of the parallel filesystem is to maintain the logical partitions and provide efficient access to data. Then we have I/O middleware on top of the parallel filesystem. It organizes access from many processes by optimizing two-phase I/O, disk I/O and data flow over the network and also provides data sieving by converting many small non-contiguous I/O requests into fewer/bigger requests. Then there would be a high-end I/O library such as HDF5, NetCDF and so on. What it does is that it maps application abstractions to storage abstractions I/O in terms of the data structures of the code. So, data is stored directly to the disk by calling this library and this library is implemented to work quite efficiently. It is better to use this kind of libraries since we support both of HDF5 and NetCDF. You could also use I/O middleware which is MPI-IO. In today’s talk, I will focus more on MPI-IO which is a part of MPI-2. However, I will also discuss the pros and cons of different approaches. And then, as you may see, there is the application which is mostly your program and your program will decide whether to use high-end I/O library or I/O middleware.

=Parallel filesystem=

On the national systems, we do have a parallel filesystem designed to scale to tens of thousand of computing nodes efficiently. For better performance, files can be striped across multiple drives. It means file does not reside on a single hard drive but multiple drives so that while a hard drive taking reading operation and another drive can send back the data to the program.

In order to avoid that two or more different processes access to a same file, parallel file systems use locks to manage this kind of concurrent file access. What actually happens is that the Files are pieced into ‘lock’ units and scattered across multiple hard drives. Then, Client nodes which is computing node obtain locks on units that they access before I/O occurs

* Files can be striped across multiple drives for better performance
* Locks used to manage concurrent file across in most parallel file system
**Files are pieced into ‘lock’ units (scattered across many drives)
**Client nodes obtain locks on units that they access before I/O occurs
**Enables caching on clients
**Locks are reclaimed from clients when others desire access

The most important part we should know is that the parallel filesystem is optimized for storing large shared files which can be possibly accessible from many computing nodes. So, it shows very poor performance to store many small size files.  As you may get told in our new user seminar, we strongly recommend users not to generate millions of small size files.

Also, how you read and write, your file format, the number of files in a directory, and how often you ls command, affects every user! Quite often we got a ticket reporting that the user cannot even ‘ls’ in his or her /work directory. Most of cases for this situation are caused by a user doing very high I/O activities in the directory and it obviously makes system slower.
 The file system is shared over the ethernet network on a cluster: hammering the file system can hurt process communications which mostly related to MPI communication. That also affects others too.

Please note that the file systems are not infinite: bandwidth, IOPs, number of files, space, . .

*Optimized for large shared files
*Poor performance under many small reads/writes (high IOPs): Do not store millions of small files
*Your use of it affects everybody! (Different from case with CPU and RAM which are not shared)
*Critical factors: how you read / write, file format, # of files in a directory and how often per sec
*File system is shared over the ethernet network on a cluster: heavy I/O can prevent the processes from communication
*File systems are LIMITED: bandwidth, IOPs, # of files, space and etc.

==Best Practices for I/O==

What would be best practices for I/O.

First of all, it is always recommended to make a plan for your data needs such as how much will be generated and how much do you need to save and where to keep.

On the national systems, different file systems (home, project, scratch) have different quotas. Scratch data is also subject to expiry. For more details see this document. Takes these limits into account before submitting a job.

And please minimize use of ‘ls’ or ‘du’ command especially in a directory with many files.

Regularly check your disk usage with quota command. Furthermore, please take warning signs that should prompt careful consideration when you have more than 100,000 files in your space and average data file size less than 100MB (if writing lots of data)

Please do ‘housekeeping’ regularly to maintain a reasonable number of file and quota. Gzip and tar command are very popular to compress multiple files and group them. So, you could reduce the number of files using these commands.

*Make a plan for your data needs: How much will you generate? How much do you need to save? Where will you keep it?
*Monitor and control usage: Minimize use of filesystem commands like ‘ls’ and ‘du’ in large directories
*Check your disk usage regularly with ‘quota’
*Warning!!
**more than 100K files in your space
**average data file size less than 100 MB for large output
*Do ‘housekeeping’ (gzip, tar, delete) regularly

=Data Formats=

===ASCII===
First of all, there is a ASCII or someone refers it as ‘text’ format. It is a human readable file format but not efficient. So, it’s good for a small input or parameter file to run a code. The ASCII format takes larger amount of storage than other types of formats and automatically it costs more for read/write operation. You could check your code implementation if you could find ‘fprintf’ in C code or open command with ‘formatted’ option in FORTRAN code.

ASCII = American Standard Code for Information Interchange
*pros: human readable, portable (architecture independent)
*cons: inefficient storage  (13 bytes per single precision float,    22 bytes per double precision,   plus delimiters), expensive for read/write

 fprintf() in C
 open(6,file=’test’,form=’formatted’);write(6,*) in F90

===Binary===

Binary format is much ‘cheaper’ in computational sense than ASCII. ASCII has 13 for single precision and 22 for double precision. The table shows an experiment in writing 128M doubles into different locations; /scratch and /tmp on GPCS system in SciNet. As you can see, it is apparent that binary writing takes way shorter time than ASCII format.

Format	/scratch	/tmp (disk)
ASCII 	173 s   	260 s
Binary	6 s     	20 s

pros: efficient storage  (4 bytes per single precision float,  8 bytes per double precision, no delimiters), efficient read / write
cons: have to know the format to read, portability (endians)

 fwrite() in C
 open(6,file=’test’,form=’unformatted’); write(6)in F90

===MetaData (XML)===

While the binary format works fine and efficient, sometimes there would be a need to store additional information such as number of variables in the array, dimensions and size of the array and so on. So, the metadata is a useful to describe the binary. In case of passing the binary files to someone else or some other programs, it would be very helpful to include those information and to use the meta data format. By the way, it could be also done by using the high-end libraries such as HDF5 and NetCDF.

*Encodes data about data: number and names of variables, their dimensions and sizes, endians, owner, date, links, comments, etc.

===Database===

Database data format is good for many small records. Using the database, data organizing and analysis can be greatly simplified. CHARENTE supports three different database packages. It is not quite common in the numerical simulation, though.

*very powerful and flexible storage approach
*data organization and analysis can be greatly simplified
*enhanced performance over seek / sort depending on usage
*open-sourcesoftware: SQLite(serverless), PostgreSQL, mySQL

===Standard scientific dataset libraries===

There are standard scientific dataset libraries. As mentioned in the previous slide, these libraries are very good not just storing the large-scale arrays in an efficient way but also they include data descriptions that the metadata format is good at. Moreover, the libraries provide data portability across platforms and languages which means the binaries generated in one machine can be read in other machines without a problem. The libraries store data automatically with compression. It could be extremely useful. For example, if you run a large-scale simulation and needs to store large dataset in particular with many repeating value such as zero, then the libraries can compress those repeating values efficiently so that you could save the data storage dramatically.

*HDF5 = Hierarchical Data Format
*NetCDF = Network Common Data Format
*Open standards and open-source libraries
*Provide data portability across platforms and languages
*Store data in binary with optional compression
*Include data description
*Optionally provide parallel I/O

=Serial and Parallel I/O=

In large parallel calculations your dataset is distributed across many processors/nodes. As shown in the right, for example, the calculation domain is decomposed into several work-load pieces and each node takes each allocation. Therefore, each node will compute the allocated domain and try to store the data into the disk. Unfortunately, in this case using parallel filesystem isn’t sufficient – you must organize parallel I/O yourself. It will be discussed shortly. For the file format, there are a couple of options such as a raw binary without metadata information or using high-end libraries (HDF5/NetCDF).

*In large parallel calculations your dataset is distributed across many processors/nodes
*In this case using parallel filesystem isn’t enough – you must organize parallel I/O yourself
*Data can be written as raw binary, HDF5 and NetCDF.

==Serial I/O (Single CPU)==

When you try to write your data from memory in multiple computing node to a single file on the disk, there would be a couple of approaches. The simplest approach is to set a ‘spokesperson’ to collect all of data from other members in the communication. Once the data is entirely collected using communication it writes the data into a file as a regular serial I/O. It is a really simple solution and easy to implement but there are several following problems. Firstly, the bandwidth for writing is limited by the rate of one client and it applies to the memory limit as well. Secondly, the operation time linearly increases with the amount of data or problem size, and moreover it increases with number of member processes because it will take longer time to collect all data into a single node or cpu. Therefore, this type of approach cannot scale.

 Pros:
 trivially simple for small I/O
 some I/O libraries not parallel
 Cons:
 bandwidth limited by the rate one client can sustain
 may not have enough memory on a node to hold all data
 won’t scale (built-in bottleneck)

==Serial I/O (N processors)==

What you can do instead is to organize each participating process to do a serial I/O. In other words, all processes perform I/O to individual files. It is somewhat efficient than the previous model but up to a certain limit.

Firstly, when you have a lot of data, you will end up with many files. One file per processors. If you run a large-calculation with many iterations with many variables, even single simulation run could generate over a thousand output files. In this case, as we discussed before, the parallel filesystem performs poor. Again we reviewed I/O best practices and hundreds of thousand files are strongly prohibited.

Secondly, output data often has to be post-processed into a file. It is additional step and it would be quite inefficient surely. Furthermore, when each processors tries to access to the disk about same time uncoordinated I/O may swamp the filesystem (file locks!)

 Pros:
 no interprocess communication or coordination necessary
 possibly better scaling than single sequential I/O
 Cons:
 as process counts increase, lots of (small) files, won’t scale
 data often must be post-processed into one file
 uncoordinated I/O may swamp the filesystem (file locks!)

==Parallel I/O (N processe to/from 1 file)==

The best approach is to do an appropriate parallel I/O. So then each participating process write the data simultaneously into a single file using the parallel I/O. The only thing you should be aware of is that you may want to do this parallel I/O in a coordinated fashion. Otherwise, it will swamp the filesystem.

 Pros:
 only one file (good for visualization, data management, storage)
 data can be stored canonically
 avoiding post-processing will scale if done correctly
 Cons:
 uncoordinated I/O will swamp the filesystem (file locks!)
 requires more design and thought

==Parallel I/O should be collective!==

For example, parallel middleware such as MPI-IO has a few different types of coordinated or uncoordinated writing options. Once coordinated writing like collective I/O is called, then the parallel middleware will know which processes and disks will get involved. Then, the parallel middleware will find an optimized operations in lower software layers for better efficiency.

*Independent I/O operations specify only what a single process will do
*Collective I/O is coordinated access to storage by a group of processes
*Functions are called by all processes participating in I/O
*Allows filesystem to know more about access as a whole, more optimization in lower software layers, better performance

=Parallel I/O techniques=

It is a part of MPI-2 standard. So, MPI-IO is good for writing a raw binary file. As you may read in this slide, the high-end libraries such as HDF5 and NetCDF are built on top of MPI-IO. Therefore, you should have MPI-IO anyway.

*MPI-IO: parallel I/O part of the MPI-2 standard (1996)
*HDF5 (Hierarchical Data Format), built on top of MPI-IO
*Parallel NetCDF (Network Common Data Format), built on top of MPI-IO

==MPI-IO==

MPI-IO is available on our systems as a default module, OpenMPI.  MPI-IO exploits analogies with MPI, writing/reading to/from file would be very similar to MPI send/receive practice if you have some experience with MPI. For example, file access is grouped via communicator in MPI. The communicator is a group for message passing in MPI. User defined MPI datatypes are also available.

*Part of the MPI-2 standard
*ROMIO is the implementation of MPI-IO in OpenMPI (default on our systems), MPICH2
*Really only widely available scientific computing parallel I/O middleware
*MPI-IO exploits analogies with MPI
**writing , sending message
**reading , receiving message
**file access grouped via communicator: collective operations
**user defined MPI datatypes, e.g. for noncontiguous data layout
**all functionality through function calls

===Basic MPI-IO operations in C===

 int MPI_File_open ( MPI_Comm comm, char* filename, int amode, MPI_Info info, MPI_File* fh)
int MPI_File_seek ( MPI_File fh, MPI_Offset offset, int to)
- updates individual file pointer
int MPI_File_set_view ( MPI_File fh, MPI_Offset offset, MPI_Datatype etype, MPI_Datatype filetype, char* datarep, MPI_Info info)
- changes process’s view of data in file ,
- etype is the elementary datatype
 int MPI_File_read ( MPI_File fh, void* buf, int count, MPI_Datatype datatype, MPI_Status* status)
int MPI_File_write (MPI_File fh, void* buf, int count,  MPI_Datatype datatype, MPI_Status* status)
int MPI_File_close ( MPI_File* fh)

Here is a simple skeleton for MPI-IO operations in C. Like a MPI code, it does have MPI_File_open and close at the beginning and at the end. There are File_write and File_read. And also, there is MPI_File_seek which used to update individual file pointer. This will be discussed in detail shortly.

MPI_File_set_view is to assign regions of the file to separate processes.
File views are specified using a triplet - (displacement, etype, and  filetype) – that is passed to MPI_File_set_view

* displacement = number of bytes to skip from the start of the file
*etype = unit of data access (can be any basic or derived datatype)
*filetype = specifies which portion of the file is visible to the process  

===Basic MPI-IO operations in F90===

MPI_FILE_OPEN (integer comm, character[] filename, integer amode, integer info, integer fh, integer ierr)
MPI_FILE_SEEK (integer fh, integer(kind=MPI_OFFSET_KIND) offset,  integer whence, integer ierr)
- updates individual file pointer
MPI_FILE_SET_VIEW (integer fh, integer(kind=MPI_OFFSET_KIND) offset,  integer etype, integer filetype,                                           character[] datarep, integer info, integer ierr)
- changes process’s view of data in file
- etype is the elementary datatype
MPI_FILE_READ (integer fh, type buf, integer count, integer datatype,  integer[MPI_STATUS_SIZE] status, integer ierr)
MPI_FILE_WRITE (integer fh, type buf, integer count, integer datatype,  integer[MPI_STATUS_SIZE] status, integer ierr)
MPI_FILE_CLOSE (integer fh)

===Opening a file requires a ...===

Opening file requires a communicator, file name, and file hand for all future reference to file. And also, it requires file access mode ‘amode’. There are a couple of different modes like MPI_MODE_WRONLY means write only. You can combine it using bitwise or “|” in C or addition “+” in FORTRAN

*Communicator
*File name
*File handle, for all future reference to file
*File access mode ‘amode’, made up of combinations of:

MPI_MODE_RDONLY                         Read only
MPI_MODE_RDWR                           Read and writing
MPI_MODE_WRONLY                         Write only
MPI_MODE_CREATE                         Create file if it does not exist
MPI_MODE_EXCL                            Error if creating file that exists
MPI_MODE_DELETE_ON_CLOSE             Delete file on close
MPI_MODE_UNIQUE_OPEN                 File not to be opened elsewhere
MPI_MODE_SEQUENTIAL                 File to be accessed sequentially
MPI_MODE_APPEND                         Position all file pointers to end

*Combine it using bitwise or “|” in C or addition “+” in FORTRAN
*Info argument usually set to ‘MPI_INFO_NULL’

===C example===

MPI_FILE fh ; MPI_File_open (MPI_COMM_WORLD, "test.dat" ,MPI_MODE_RDONLY,  MPI_INFO_NULL,&fh );
... read some data here ...
MPI_File_close(&fh ) ;

===F90 example===

integer :: fh,ierr call MPI_FILE_OPEN(MPI_COMM_WORLD,"test.dat", MPI_MODE_RDONLY, MPI_INFO_NULL, fh, ierr)
... read some data here ...
call MPI_FILE_CLOSE(fh, ierr )

===Read/Write contiguous data===
So, let us think to write one file from four different processes. As shown in the figure, each process will write its data into a designated portion in the same file. Writing proceeds in a contiguous fashion from process 0 to 3.

===example in C===
Basically we initialize MPI and initialize several variable arrays. Using MPI_Comm_rank, each process will have its own rank or process id. Using for (i=0), array a is set as it’s rank for 10 array size.  for example, on process 3 creates a array for 10 of 3 characters.

 MPI_File_open (MPI_COMM_WORLD, “data.out" , MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);

We defined the communicator and filename ‘data.out’, for the mode, we combined ‘write only’ and ‘create file if it does not exist’. And then, we define the offset where each process starts to write. As you can see, process 0 starts from the beginning and process 1 is next in a contiguous fashion.

 MPI_Offset displace = rank*n*sizeof(char);

So, the offset will be calculated by multiplying rank*size of array*sizeof (char). Now we are ready to assign the writing regions to each process using MPI_File_set_view. displacement is set, etype and filetype are set as ‘MPI_CHAR’. ‘native’ means that data in this representation is stored in a file exactly as it is in memory. And then, we command ‘write’ using MPI_File_write.

#include  
#include
 int main(int argc, char **argv) {

int rank, i; char a[10];
 MPI_Offset n = 10; MPI_File fh ; MPI_Status status ;

MPI_Init(&argc, &argv);

MPI_Comm_rank(MPI_COMM_WORLD, &rank);

for (i=0; i<10; i++)
 a[i] = (char)( ’0’ + rank);  // e.g. on processor 3 creates a[0:9]=’3333333333’

MPI_File_open (MPI_COMM_WORLD, “data.out", MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
MPI_Offset displace = rank*n*sizeof(char); // start of the view for each processor

MPI_File_set_view (fh , displace , MPI_CHAR, MPI_CHAR, "native" ,MPI_INFO_NULL);  
// note that etype and filetype are the same

MPI_File_write(fh, a, n, MPI_CHAR, &status);

MPI_File_close(&fh ) ;

MPI_Finalize ( ) ;

return 0;

===Summary: MPI-IO===

As you may notice, its implementation seems quite straight-forward. There must be much of advanced materials using MPI-IO but it seems a bit beyond the scope of this seminar. So, in summary, MPI-IO is a part of standard MPI-2 library, and it is a very widely installed on almost all of HPC systems with modern MPI versions. We installed OpenMPI which supports MPI-IO on all of our clusters. MPI-IO doesn’t require to install additional libraries but unfortunately it writes raw data to file. So, it is not portable across platforms, hard to append new variables and doesn’t include data description.

==NetCDF==

Network Common Data For

NetCDF is one of most popular packages in storing data. Basically, NetCDF covers up all of what MPI-IO cannot support. It uses MPI-IO under the hood but instead of specifying the offset you just need to call NetCDF and tell what arrays you want to store. Then, NetCDF will handle it and try to store it in a contiguous fashion. In NetCDF Data stored as binary and, as mentioned before, it supports Self-describing, metadata in the header and Portable across different architectures and
Optional compression. One of better points comparing to HDF5, NetCDF supports a variety of visualization packages such as Paraview. We have both serial and parallel NetCDF on our systems.

*Format for storing large arrays, uses MPI-IO under the hood
*Libraries for C/C++, Fortran 77/90/95/2003, Python, Java, R, Ruby, etc.
*Data stored as binary
*Self-describing, metadata in the header (can be queried by utilities)
*Portable across different architectures
*Optional compression
*Uses MPI-IO, optimized for performance
*Parallel NetCDF

===example in C===
Basically we initialize MPI and initialize several variable arrays. Using MPI_Comm_rank, each process will have its own rank or process id. Using for (i=0), array a is set as it’s rank for 10 array size.  for example, on process 3 creates a array for 10 of 3 characters.

 MPI_File_open (MPI_COMM_WORLD, “data.out" , MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);

We defined the communicator and filename ‘data.out’, for the mode, we combined ‘write only’ and ‘create file if it does not exist’. And then, we define the offset where each process starts to write. As you can see, process 0 starts from the beginning and process 1 is next in a contiguous fashion.

 MPI_Offset displace = rank*n*sizeof(char);

So, the offset will be calculated by multiplying rank*size of array*sizeof (char). Now we are ready to assign the writing regions to each process using MPI_File_set_view. displacement is set, etype and filetype are set as ‘MPI_CHAR’. ‘native’ means that data in this representation is stored in a file exactly as it is in memory. And then, we command ‘write’ using MPI_File_write. Once compiled and run successfully, you can have the output as shown in a file.

#include
#include
#include
#define FILE_NAME "simple_xy.nc" #define NDIMS 2
#define NX 3
#define NY 4
int main() {
int ncid, x_dimid, y_dimid, varid; int dimids[NDIMS];
int data_out[NX][NY];
int x, y, retval;
for (x = 0; x < NX; x++)
     for (y = 0; y < NY; y++)
          data_out[x][y] = x * NY + y;
retval = nc_create(FILE_NAME, NC_CLOBBER, &ncid);
retval = nc_def_dim(ncid, "x", NX, &x_dimid);
retval = nc_def_dim(ncid, "y", NY, &y_dimid);
dimids[0] = x_dimid;
dimids[1] = y_dimid;
retval = nc_def_var(ncid, "data", NC_INT, NDIMS, dimids, &varid);
retval = nc_enddef(ncid);
retval = nc_put_var_int(ncid, varid, &data_out[0][0]);
retval = nc_close(ncid);
return 0;
}

==HDF5==

Hierarchical Data Format

HDF5 is also very popular tool in storing data. It supports most of NetCDF features such Self-describing file format for large datasets, and also uses MPI-IO under the hood. Basically, HDF5 is More general than NetCDF, with object-oriented description of datasets, groups, attributes, types, data spaces and property lists. We have both serial and parallel HDF5 on our systems.

*Self-describing file format for large datasets, uses MPI-IO under the hood
*Libraries for C/C++, Fortran 90, Java, Python, R
*More general than NetCDF, with object-oriented description of datasets, groups, attributes, types, data spaces and property lists
*File content can be arranged into a Unix-like filesystem /path/to/resource
**data sets containing homogeneous multidimensional images/tables/arrays
**groups containing structures which can hold datasets and other groups
*Header information can be queried by utilities
*Optional compression (good for arrays with many similar elements)
*We provide both serial and parallel HDF5

=References=
*https://www.nhr.kit.edu/userdocs/horeka/parallel_IO/
*https://hpc-forge.cineca.it/files/CoursesDev/public/2017/Parallel_IO_and_management_of_large_scientific_data/Roma/MPI-IO_2017.pdf
*https://janth.home.xs4all.nl/MPIcourse/PDF/08_MPI_IO.pdf
*https://events.prace-ri.eu/event/176/contributions/59/attachments/170/326/Advanced_MPI_II.pdf
*https://www.cscs.ch/fileadmin/user_upload/contents_publications/tutorials/fast_parallel_IO/MPI-IO_NS.pdf