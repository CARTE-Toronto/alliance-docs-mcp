---
title: "OpenACC Tutorial - Data movement/en"
url: "https://docs.alliancecan.ca/wiki/OpenACC_Tutorial_-_Data_movement/en"
category: "User Guide"
last_modified: "2017-05-12T18:27:08Z"
page_id: 1479
display_title: "OpenACC Tutorial - Data movement"
---

`<languages />`{=html}

## Making Data Management Explicit {#making_data_management_explicit}

![Data management with Unified Memory\|right ](https://docs.alliancecan.ca/Intro-DM.png "Data management with Unified Memory|right "){width="200"} We used CUDA Unified Memory to simplify the first steps in accelerating our code. This made the process simple, but it also made the code not portable:

- PGI-only: --ta=tesla:managed flag
- NVIDIA-only: CUDA Unified Memory

Explicitly managing data will make the code portable and may improve performance.

## Structured Data Regions {#structured_data_regions}

The `data` directive defines a region of code in which GPU arrays remain on the GPU and are shared among all kernels in that region. Here is an example of defining the structured data region with data directives:

``` {.cpp .numberLines}
#pragma acc data
{
#pragma acc parallel loop ...
#pragma acc parallel loop
...
}
```

another example:

``` {.cpp .numberLines}
!$acc data
!$acc parallel loop
...
!$acc parallel loop
...
!$acc end data
```

## Unstructured Data {#unstructured_data}

We sometimes encounter a situation where scoping does not allow the use of normal data regions (e.g. when using constructors or destructors).

### Directives

In those cases, we use unstructured data directives:

- **enter data** - defines the start of an unstructured data lifetime
  - clauses : **copyin(list)**, **create(list)**
- **exit data** - defines the end of an unstructured data lifetime
  - clauses: **copyout(list)\'\',**delete(list)\'\'\'

Below is the example of using the unstructured data directives in the code:

``` {.cpp .numberLines}
#pragma acc enter data copyin(a)
...
#pragma acc exit data delete(a)
```

### C++ Classes {#c_classes}

What is the advantage of having unstructured data clauses? In fact, unstructured data clauses enable OpenACC to be used in C++ classes. Moreover, unstructured data clauses can be used whenever data is allocated and initialized in a different piece of code than where it is freed (e.g. Fortran modules).

``` {.cpp .numberLines}
class Matrix { Matrix(int n) {
len = n;
v = new double[len];
#pragma acc enter data
                     create(v[0:len])
}
~Matrix() {
#pragma acc exit data 
                     delete(v[0:len])
};
```

### Data Clauses {#data_clauses}

- **copyin(list)** - Allocates memory on GPU and copies data from host to GPU when entering region.
- **copyout(list)** - Allocates memory on GPU and copies data to the host when exiting region.
- **copy(list)** - Allocates memory on GPU and copies data from host to GPU when entering region and copies data to the host when exiting region. (Structured Only)
- **create(list)** - Allocates memory on GPU but does not copy.
- **delete(list)** - Deallocate memory on the GPU without copying. (Unstructured Only)
- **present(list)** - Data is already present on GPU from another containing data region.

### Array Shape {#array_shape}

Sometimes, the compiler cannot determine the size of arrays. Therefore, one have to explicitly specify the size with data clauses and array \"shape\". Here is an example of array shape in C:

``` {.cpp .numberLines}
#pragma acc data copyin(a[0:nelem]) copyout(b[s/4:3*s/4])
```

In Fortran it will look like this:

``` {.cpp .numberLines}
!$acc data copyin(a(1:end)) copyout(b(s/4:3*s/4))
```

## Explicit Data Movement: Examples {#explicit_data_movement_examples}

### Copy In Matrix {#copy_in_matrix}

In the following example we allocate and initialize the matrix first. Then copy it to the device. The copy procedure is done in two steps:

1.  Copy the structure of the matrix.
2.  Copy its members.

``` {.cpp .numberLines}
void allocate_3d_poisson_matrix(matrix &A, int N) { 
   int num_rows=(N+1)*(N+1)*(N+1);
   int nnz=27*num_rows;
   A.num_rows=num_rows;
   A.row_offsets = (unsigned int*) \ malloc((num_rows+1)*sizeof(unsigned int));
   A.cols = (unsigned int*)malloc(nnz*sizeof(unsigned int));
   A.coefs = (double*)malloc(nnz*sizeof(double)); // Initialize Matrix
   A.row_offsets[num_rows]=nnz;
   A.nnz=nnz;
   #pragma acc enter data copyin(A)
   #pragma acc enter data copyin(A.row_offsets[:num_rows+1],A.cols[:nnz],A.coefs[:nnz])
}
```

### Delete Matrix {#delete_matrix}

In this example, in order to free the device memory we first remove the matrix from the device, then issue the free instructions. Again, this is done in two steps (in reverse order):

1.  Delete the members.
2.  Delete the structure.

``` {.cpp .numberLines}
void free_matrix(matrix &A) {
   unsigned int *row_offsets=A.row_offsets; 
   unsigned int * cols=A.cols;
   double * coefs=A.coefs;
   #pragma acc exit data delete(A.row_offsets,A.cols,A.coefs) 
   #pragma acc exit data delete(A)
   free(row_offsets); 
   free(cols); 
   free(coefs);
}
```

### The `present` Clause {#the_present_clause}

When managing the memory at a higher level, it's necessary to inform the compiler that data is already present on the device. Local variables should still be declared in the function where they're used.

``` {.cpp .numberLines}
function main(int argc, char **argv) {
#pragma acc data copy(A) {
    laplace2D(A,n,m);
}
}
...
function laplace2D(double[N][M] A,n,m){
   #pragma acc data present(A[n][m]) create(Anew)
   while ( err > tol && iter < iter_max ) {
      err=0.0;
      ...
   }
}
```

In our next example, we inform the compiler in the compute region of the code that our data is already present:

``` {.cpp .numberLines}
#pragma acc kernels \
present(row_offsets,cols,Acoefs,xcoefs,ycoefs)
{
   for(int i=0;i<num_rows;i++) {
      double sum=0;
      int row_start=row_offsets[i];
      int row_end=row_offsets[i+1]; 
      for(int j=row_start;j<row_end;j++) {
         unsigned int Acol=cols[j]; 
         double Acoef=Acoefs[j]; 
         double xcoef=xcoefs[Acol]; 
         sum+=Acoef*xcoef;
      }
   ycoefs[i]=sum; 
   }
}
```

### Compiling and Running with Explicit Memory Management {#compiling_and_running_with_explicit_memory_management}

In order to rebuild the code without managed memory change **-ta=tesla:managed** to **-ta-tesla** in the Makefile.

### Update Directive {#update_directive}

In OpenACC it is possible to specify an array (or part of an array) that should be refreshed within the data region. In order to do so we use the `update` directive:

``` {.cpp .numberLines}
do_something_on_device()
!$acc update self(a)   //  '''Copy "a" from GPU to CPU'''
do_something_on_host()
!$acc update device(a)  // '''Copy "a" from CPU to GPU'''
```

In the following example we demonstrate the usage of the `update` directive. First, we modify a vector on the CPU (host), then copy it to the GPU (device):

``` {.cpp .numberLines}
void initialize_vector(vector &v,double val) {
   for(int i=0;i<v.n;i++) 
      v.coefs[i]=val;   // '''Updating the vector on the CPU '''
   #pragma acc update 
      device(v.coefs[:v.n])    // '''Updating the vector on the GPU'''
}
```

### Build and Run without Managed Memory {#build_and_run_without_managed_memory}

Below we demonstrate the performance of the code with and without managed memory. ![Benchmarking codes with and without managed memory\|right ](https://docs.alliancecan.ca/Benchmark-DM.png "Benchmarking codes with and without managed memory|right "){width="300"} In this example, several benchmarks that use OpenACC directives have been compiled with and without the **-ta=tesla:managed** option: ![Another benchark results \|right ](https://docs.alliancecan.ca/Benchmark2-DM.png "Another benchark results |right "){width="300"} The results indicate that some of these tests improve speed using managed memory, but probably due to the use of the pinned memory in the data transfers. Overall, the results suggest that data locality indeed works: when most of the operations are on the GPU and data stays on the GPU for a long time, the data movement does not play a significant role in the performance.

[Onward to the next unit: Optimizing loops](https://docs.alliancecan.ca/OpenACC_Tutorial_-_Optimizing_loops "Onward to the next unit: Optimizing loops"){.wikilink}\
[Back to the lesson plan](https://docs.alliancecan.ca/OpenACC_Tutorial "Back to the lesson plan"){.wikilink}
