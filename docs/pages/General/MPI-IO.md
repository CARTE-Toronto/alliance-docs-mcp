---
title: "MPI-IO/en"
url: "https://docs.alliancecan.ca/wiki/MPI-IO/en"
category: "General"
last_modified: "2021-07-05T22:33:10Z"
page_id: 12992
display_title: "MPI-IO"
---

== Description ==
MPI-IO is a family of MPI routines that makes it possible to do file read and write operations in parallel. MPI-IO is a part of the MPI-2 standard. The main advantage of MPI-IO is that it allows, in a simple and efficient fashion, to write and to read data that is partitioned on multiple processes, to and from a single file that is common to all processes. This is particularly useful when the manipulated data are vectors or matrices that are cut up in a structured manner between the different processes involved. This page gives a few guidelines on the use of MPI-IO and some references to more complete documentation.

== Using MPI-IO ==

=== Operations through offsets ===

The simplest way to perform parallel read and write operations is to use offsets. Each process can read from or write to the file with a defined offset. This can be done in two operations (MPI_File_seek followed by MPI_File_read or by MPI_File_write), or even in a single operation (MPI_File_read_at or MPI_File_write_at). Usually the offset is computed as a function of the process rank.
 MPI_MODE_CREATE), MPI_INFO_NULL, &f);

    /* Write data alternating between the processes: aabbccddaabbccdd... */
    MPI_File_seek(f, rank*BLOCKSIZE, MPI_SEEK_SET); /* Go to position rank * BLOCKSIZE */
    for (i=0; i<NBRBLOCKS; ++i) {
        MPI_File_write(f, buffer, BLOCKSIZE, MPI_CHAR, MPI_STATUS_IGNORE);
        /* Advance (size-1)*BLOCKSIZE bytes */
        MPI_File_seek(f, (size-1)*BLOCKSIZE, MPI_SEEK_CUR);
    }

    MPI_File_close(&f);

    MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &f);

    /* Read data in a serial fashion for each process. Each process reads: aabbccdd */
    for (i=0; i<NBRBLOCKS; ++i) {
        MPI_File_read_at(f, rank*i*NBRBLOCKS*BLOCKSIZE, buffer, BLOCKSIZE, MPI_CHAR, MPI_STATUS_IGNORE);
    }

    MPI_File_close(&f);
    MPI_Finalize();

    return 0;
}
}}
=== Using views ===

Using views, each process can see a section of the file as if it were the entire file. In this way it is no longer necessary to compute the file offsets as a function of the process rank. Once the view is defined, it is a lot simpler to perform operations on this file, without risking conflicts with operations performed by other processes. A view is defined using the function MPI_File_set_view. Here is a program identical to the previous one, but using views instead.
 MPI_MODE_CREATE),
        MPI_INFO_NULL,
        &f);

    /* Write data alternating between the processes: aabbccddaabbccdd... */
    MPI_Type_contiguous(BLOCKSIZE, MPI_CHAR, &type_intercomp);
    MPI_Type_commit(&type_intercomp);
    for (i=0; i<NBRBLOCKS; ++i) {
        MPI_File_set_view(f, rank*BLOCKSIZE+i*size*BLOCKSIZE, MPI_CHAR, type_intercomp, "native", MPI_INFO_NULL);
        MPI_File_write(f, buffer, BLOCKSIZE, MPI_CHAR, MPI_STATUS_IGNORE);
    }

    MPI_File_close(&f);

    MPI_File_open(MPI_COMM_WORLD,
        filename,
        MPI_MODE_RDONLY,
        MPI_INFO_NULL,
        &f);

    /* Read data in a serial fashion for each process. Each process reads: aabbccdd */
    MPI_Type_contiguous(NBRBLOCKS*BLOCKSIZE, MPI_CHAR, &type_contiguous);
    MPI_Type_commit(&type_contiguous);
    MPI_File_set_view(f, rank*NBRBLOCKS*BLOCKSIZE, MPI_CHAR, type_contiguous, "native", MPI_INFO_NULL);
    for (i=0; i<NBRBLOCKS; ++i) {
        MPI_File_read(f,  buffer, BLOCKSIZE, MPI_CHAR, MPI_STATUS_IGNORE);
    }

    MPI_File_close(&f);
    MPI_Finalize();

    return 0;
}
}}
Warning! Some file systems do not support file locks. Consequently some operations are not possible, in particular using views on disjoint file sections.

== References ==

* OpenMPI documentation
* Course on parallel I/O