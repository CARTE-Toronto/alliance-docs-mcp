---
title: "CESM"
url: "https://docs.alliancecan.ca/wiki/CESM"
category: "General"
last_modified: "2025-12-04T20:02:09Z"
page_id: 24186
display_title: "CESM"
---

\"The [Community Earth System Model](https://www.cesm.ucar.edu/) is a fully coupled global climate model developed in collaboration with colleagues in the research community. CESM provides state of the art computer simulations of Earth\'s past, present, and future climate states.\"

# Porting and Validating {#porting_and_validating}

The below configuration files and commands are designed for a local installation of CESM 2.1. Local installations allow for [source code changes](https://ncar.github.io/CESM-Tutorial/notebooks/sourcemods/sourcemods.html) which may be useful for specific research purposes. Before making the adaptations as described in the sections below, please [download CESM 2.1 from the CESM developers](https://www.cesm.ucar.edu/models/cesm2/download) in your local directory.

`<tabs>`{=html} `<tab name="Version 2.1.5">`{=html}

This version is based on the [latest official release](https://github.com/ESCOMP/CESM/releases/latest).

`</tab>`{=html} `<tab name="Version 2.1.3">`{=html}

This older version has been very popular in the research community, but it may become obsolete because this version is no longer officially supported.

To make this version work, `<b>`{=html}some external dependencies must be replaced`</b>`{=html} with newer versions which are no longer exactly matching the 2.1.3 version of CESM; researchers are responsible for confirming validity.

`</tab>`{=html} `</tabs>`{=html}

## Checkout externals {#checkout_externals}

Before your first use of CESM, you may checkout the individual model components by running the checkout_externals script.

You may need to accept a certificate from the CESM repository to download input files. To validate, run the same script with `-S`.

See [this documentation page](https://escomp.github.io/CESM/versions/cesm2.1/html/downloading_cesm.html) for an example of a valid output.

## Local machine file {#local_machine_file}

- Create and edit the file `~/.cime/config_machines.xml` from the following minimal content per cluster; `<b>`{=html}update both configuration lines`</b>`{=html} having `def-EDIT_THIS` with the compute account you want to use on the cluster.
  `<tabs>`{=html} `<tab name="Beluga">`{=html} `{{File
    |name=~/.cime/config_machines.xml
    |lang="xml"
    |contents=
  <?xml version="1.0"?>

  <config_machines version="2.0">
    <machine MACH="beluga">
      <DESC>https://docs.alliancecan.ca/wiki/Béluga/en</DESC>
      <NODENAME_REGEX>b[cegl].*.int.ets1.calculquebec.ca</NODENAME_REGEX>

      <OS>LINUX</OS>
      <COMPILERS>intel,gnu</COMPILERS>
      <MPILIBS>openmpi</MPILIBS>

      <PROJECT>def-EDIT_THIS</PROJECT>
      <CHARGE_ACCOUNT>def-EDIT_THIS</CHARGE_ACCOUNT>

      <CIME_OUTPUT_ROOT>/scratch/$USER/cesm/output</CIME_OUTPUT_ROOT>
      <DIN_LOC_ROOT>/scratch/$USER/cesm/inputdata</DIN_LOC_ROOT>
      <DIN_LOC_ROOT_CLMFORC>${DIN_LOC_ROOT}/atm/datm7</DIN_LOC_ROOT_CLMFORC>
      <DOUT_S_ROOT>$CIME_OUTPUT_ROOT/archive/case</DOUT_S_ROOT>
      <GMAKE>make</GMAKE>
      <GMAKE_J>8</GMAKE_J>
      <BATCH_SYSTEM>slurm</BATCH_SYSTEM>
      <SUPPORTED_BY>support@tech.alliancecan.ca</SUPPORTED_BY>
      <MAX_TASKS_PER_NODE>40</MAX_TASKS_PER_NODE>
      <MAX_MPITASKS_PER_NODE>40</MAX_MPITASKS_PER_NODE>
      <PROJECT_REQUIRED>TRUE</PROJECT_REQUIRED>

      <mpirun mpilib="openmpi">
        <executable>srun</executable>
      </mpirun>
      <module_system type="module" allow_error="true">
        <init_path lang="perl">/cvmfs/soft.computecanada.ca/custom/software/lmod/lmod/init/perl</init_path>
        <init_path lang="python">/cvmfs/soft.computecanada.ca/custom/software/lmod/lmod/init/env_modules_python.py</init_path>
        <init_path lang="csh">/cvmfs/soft.computecanada.ca/custom/software/lmod/lmod/init/csh</init_path>
        <init_path lang="sh">/cvmfs/soft.computecanada.ca/custom/software/lmod/lmod/init/sh</init_path>
        <cmd_path lang="perl">/cvmfs/soft.computecanada.ca/custom/software/lmod/lmod/libexec/lmod perl</cmd_path>
        <cmd_path lang="python">/cvmfs/soft.computecanada.ca/custom/software/lmod/lmod/libexec/lmod python</cmd_path>
        <cmd_path lang="csh">module</cmd_path>
        <cmd_path lang="sh">module</cmd_path>
        <modules>
        <command name="purge"/>
      <command name="load">StdEnv/2023</command>
        </modules>
        <modules compiler="intel">
      <command name="load">intel/2023.2.1</command>
      <command name="load">git-annex/10.20231129</command>
      <command name="load">cmake/3.27.7</command>
        </modules>
        <modules mpilib="openmpi">
          <command name="load">openmpi/4.1.5</command>
          <command name="load">hdf5-mpi/1.14.2</command>
          <command name="load">netcdf-c++4-mpi/4.3.1</command>
          <command name="load">netcdf-fortran-mpi/4.6.1</command>
          <command name="load">netcdf-mpi/4.9.2</command>
      <command name="load">xml-libxml/2.0208</command>
      <command name="load">flexiblas/3.3.1</command>
        </modules>
      </module_system>
      <environment_variables>
              <env name="NETCDF_PATH">/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v4/MPI/intel2023/openmpi4/pnetcdf/1.12.3</env>
              <env name="NETCDF_FORTRAN_PATH">/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v4/MPI/intel2023/openmpi4/netcdf-fortran-mpi/4.6.1/</env>
              <env name="NETCDF_C_PATH">/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v4/MPI/intel2023/openmpi4/netcdf-c++4-mpi/4.3.1/</env>
              <env name="NETLIB_LAPACK_PATH">/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Core/imkl/2023.2.0/mkl/2023.2.0/</env>
          <env name="OMP_STACKSIZE">256M</env>
              <env name="I_MPI_CC">icc</env>
              <env name="I_MPI_FC">ifort</env>
              <env name="I_MPI_F77">ifort</env>
              <env name="I_MPI_F90">ifort</env>
              <env name="I_MPI_CXX">icpc</env>
      </environment_variables>
      <resource_limits>
        <resource name="RLIMIT_STACK">300000000</resource>
      </resource_limits>
    </machine>
  </config_machines>
  }}`{=mediawiki} `</tab>`{=html}

  `<tab name="Cedar">`{=html} `{{File
    |name=~/.cime/config_machines.xml
    |lang="xml"
    |contents=
  <?xml version="1.0"?>

  <config_machines version="2.0">
    <machine MACH="cedar">
      <DESC>https://docs.alliancecan.ca/wiki/Cedar</DESC>
      <NODENAME_REGEX>c[de].*.computecanada.ca</NODENAME_REGEX>

      <OS>LINUX</OS>
      <COMPILERS>intel,gnu</COMPILERS>
      <MPILIBS>openmpi</MPILIBS>

      <PROJECT>def-EDIT_THIS</PROJECT>
      <CHARGE_ACCOUNT>def-EDIT_THIS</CHARGE_ACCOUNT>

      <CIME_OUTPUT_ROOT>/scratch/$USER/cesm/output</CIME_OUTPUT_ROOT>
      <DIN_LOC_ROOT>/scratch/$USER/cesm/inputdata</DIN_LOC_ROOT>
      <DIN_LOC_ROOT_CLMFORC>${DIN_LOC_ROOT}/atm/datm7</DIN_LOC_ROOT_CLMFORC>
      <DOUT_S_ROOT>$CIME_OUTPUT_ROOT/archive/case</DOUT_S_ROOT>
      <GMAKE>make</GMAKE>
      <GMAKE_J>8</GMAKE_J>
      <BATCH_SYSTEM>slurm</BATCH_SYSTEM>
      <SUPPORTED_BY>support@tech.alliancecan.ca</SUPPORTED_BY>
      <MAX_TASKS_PER_NODE>48</MAX_TASKS_PER_NODE>
      <MAX_MPITASKS_PER_NODE>48</MAX_MPITASKS_PER_NODE>
      <PROJECT_REQUIRED>TRUE</PROJECT_REQUIRED>

      <mpirun mpilib="openmpi">
        <executable>srun</executable>
      </mpirun>
      <module_system type="module" allow_error="true">
        <init_path lang="perl">/cvmfs/soft.computecanada.ca/custom/software/lmod/lmod/init/perl</init_path>
        <init_path lang="python">/cvmfs/soft.computecanada.ca/custom/software/lmod/lmod/init/env_modules_python.py</init_path>
        <init_path lang="csh">/cvmfs/soft.computecanada.ca/custom/software/lmod/lmod/init/csh</init_path>
        <init_path lang="sh">/cvmfs/soft.computecanada.ca/custom/software/lmod/lmod/init/sh</init_path>
        <cmd_path lang="perl">/cvmfs/soft.computecanada.ca/custom/software/lmod/lmod/libexec/lmod perl</cmd_path>
        <cmd_path lang="python">/cvmfs/soft.computecanada.ca/custom/software/lmod/lmod/libexec/lmod python</cmd_path>
        <cmd_path lang="csh">module</cmd_path>
        <cmd_path lang="sh">module</cmd_path>
        <modules>
        <command name="purge"/>
      <command name="load">StdEnv/2023</command>
        </modules>
        <modules compiler="intel">
      <command name="load">intel/2023.2.1</command>
      <command name="load">git-annex/10.20231129</command>
      <command name="load">cmake/3.27.7</command>
        </modules>
        <modules mpilib="openmpi">
          <command name="load">openmpi/4.1.5</command>
          <command name="load">hdf5-mpi/1.14.2</command>
          <command name="load">netcdf-c++4-mpi/4.3.1</command>
          <command name="load">netcdf-fortran-mpi/4.6.1</command>
          <command name="load">netcdf-mpi/4.9.2</command>
      <command name="load">xml-libxml/2.0208</command>
      <command name="load">flexiblas/3.3.1</command>
        </modules>
      </module_system>
      <environment_variables>
              <env name="NETCDF_PATH">/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v4/MPI/intel2023/openmpi4/pnetcdf/1.12.3</env>
              <env name="NETCDF_FORTRAN_PATH">/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v4/MPI/intel2023/openmpi4/netcdf-fortran-mpi/4.6.1/</env>
              <env name="NETCDF_C_PATH">/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v4/MPI/intel2023/openmpi4/netcdf-c++4-mpi/4.3.1/</env>
              <env name="NETLIB_LAPACK_PATH">/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Core/imkl/2023.2.0/mkl/2023.2.0/</env>
          <env name="OMP_STACKSIZE">256M</env>
              <env name="I_MPI_CC">icc</env>
              <env name="I_MPI_FC">ifort</env>
              <env name="I_MPI_F77">ifort</env>
              <env name="I_MPI_F90">ifort</env>
              <env name="I_MPI_CXX">icpc</env>
      </environment_variables>
      <resource_limits>
        <resource name="RLIMIT_STACK">300000000</resource>
      </resource_limits>
    </machine>
  </config_machines>
  }}`{=mediawiki} `</tab>`{=html}

  `<tab name="Graham">`{=html} `{{File
    |name=~/.cime/config_machines.xml
    |lang="xml"
    |contents=
  <?xml version="1.0"?>

  <config_machines version="2.0">
    <machine MACH="graham">
      <DESC>https://docs.alliancecan.ca/wiki/Graham</DESC>
      <NODENAME_REGEX>gra.*</NODENAME_REGEX>

      <OS>LINUX</OS>
      <COMPILERS>intel,gnu</COMPILERS>
      <MPILIBS>openmpi</MPILIBS>

      <PROJECT>def-EDIT_THIS</PROJECT>
      <CHARGE_ACCOUNT>def-EDIT_THIS</CHARGE_ACCOUNT>

      <CIME_OUTPUT_ROOT>/scratch/$USER/cesm/output</CIME_OUTPUT_ROOT>
      <DIN_LOC_ROOT>/scratch/$USER/cesm/inputdata</DIN_LOC_ROOT>
      <DIN_LOC_ROOT_CLMFORC>${DIN_LOC_ROOT}/atm/datm7</DIN_LOC_ROOT_CLMFORC>
      <DOUT_S_ROOT>$CIME_OUTPUT_ROOT/archive/case</DOUT_S_ROOT>
      <GMAKE>make</GMAKE>
      <GMAKE_J>8</GMAKE_J>
      <BATCH_SYSTEM>slurm</BATCH_SYSTEM>
      <SUPPORTED_BY>support@tech.alliancecan.ca</SUPPORTED_BY>
      <MAX_TASKS_PER_NODE>44</MAX_TASKS_PER_NODE>
      <MAX_MPITASKS_PER_NODE>44</MAX_MPITASKS_PER_NODE>
      <PROJECT_REQUIRED>TRUE</PROJECT_REQUIRED>

      <mpirun mpilib="openmpi">
        <executable>srun</executable>
      </mpirun>
      <module_system type="module" allow_error="true">
        <init_path lang="perl">/cvmfs/soft.computecanada.ca/custom/software/lmod/lmod/init/perl</init_path>
        <init_path lang="python">/cvmfs/soft.computecanada.ca/custom/software/lmod/lmod/init/env_modules_python.py</init_path>
        <init_path lang="csh">/cvmfs/soft.computecanada.ca/custom/software/lmod/lmod/init/csh</init_path>
        <init_path lang="sh">/cvmfs/soft.computecanada.ca/custom/software/lmod/lmod/init/sh</init_path>
        <cmd_path lang="perl">/cvmfs/soft.computecanada.ca/custom/software/lmod/lmod/libexec/lmod perl</cmd_path>
        <cmd_path lang="python">/cvmfs/soft.computecanada.ca/custom/software/lmod/lmod/libexec/lmod python</cmd_path>
        <cmd_path lang="csh">module</cmd_path>
        <cmd_path lang="sh">module</cmd_path>
        <modules>
        <command name="purge"/>
      <command name="load">StdEnv/2023</command>
        </modules>
        <modules compiler="intel">
      <command name="load">intel/2023.2.1</command>
      <command name="load">git-annex/10.20231129</command>
      <command name="load">cmake/3.27.7</command>
        </modules>
        <modules mpilib="openmpi">
          <command name="load">openmpi/4.1.5</command>
          <command name="load">hdf5-mpi/1.14.2</command>
          <command name="load">netcdf-c++4-mpi/4.3.1</command>
          <command name="load">netcdf-fortran-mpi/4.6.1</command>
          <command name="load">netcdf-mpi/4.9.2</command>
      <command name="load">xml-libxml/2.0208</command>
      <command name="load">flexiblas/3.3.1</command>
        </modules>
      </module_system>
      <environment_variables>
              <env name="NETCDF_PATH">/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v4/MPI/intel2023/openmpi4/pnetcdf/1.12.3</env>
              <env name="NETCDF_FORTRAN_PATH">/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v4/MPI/intel2023/openmpi4/netcdf-fortran-mpi/4.6.1/</env>
              <env name="NETCDF_C_PATH">/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v4/MPI/intel2023/openmpi4/netcdf-c++4-mpi/4.3.1/</env>
              <env name="NETLIB_LAPACK_PATH">/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Core/imkl/2023.2.0/mkl/2023.2.0/</env>
          <env name="OMP_STACKSIZE">256M</env>
              <env name="I_MPI_CC">icc</env>
              <env name="I_MPI_FC">ifort</env>
              <env name="I_MPI_F77">ifort</env>
              <env name="I_MPI_F90">ifort</env>
              <env name="I_MPI_CXX">icpc</env>
      </environment_variables>
      <resource_limits>
        <resource name="RLIMIT_STACK">300000000</resource>
      </resource_limits>
    </machine>
  </config_machines>
  }}`{=mediawiki} `</tab>`{=html}

  `<tab name="Narval">`{=html} Note: despite the Intel software dependencies, the below configuration works on Narval\'s AMD processors. `{{File
    |name=~/.cime/config_machines.xml
    |lang="xml"
    |contents=
  <?xml version="1.0"?>

  <config_machines version="2.0">
    <machine MACH="narval">
      <DESC>https://docs.alliancecan.ca/wiki/Narval/en</DESC>
      <NODENAME_REGEX>n[acgl].*.narval.calcul.quebec</NODENAME_REGEX>

      <OS>LINUX</OS>
      <COMPILERS>intel,gnu</COMPILERS>
      <MPILIBS>openmpi</MPILIBS>

      <PROJECT>def-EDIT_THIS</PROJECT>
      <CHARGE_ACCOUNT>def-EDIT_THIS</CHARGE_ACCOUNT>

      <CIME_OUTPUT_ROOT>/scratch/$USER/cesm/output</CIME_OUTPUT_ROOT>
      <DIN_LOC_ROOT>/scratch/$USER/cesm/inputdata</DIN_LOC_ROOT>
      <DIN_LOC_ROOT_CLMFORC>${DIN_LOC_ROOT}/atm/datm7</DIN_LOC_ROOT_CLMFORC>
      <DOUT_S_ROOT>$CIME_OUTPUT_ROOT/archive/case</DOUT_S_ROOT>
      <GMAKE>make</GMAKE>
      <GMAKE_J>8</GMAKE_J>
      <BATCH_SYSTEM>slurm</BATCH_SYSTEM>
      <SUPPORTED_BY>support@tech.alliancecan.ca</SUPPORTED_BY>
      <MAX_TASKS_PER_NODE>64</MAX_TASKS_PER_NODE>
      <MAX_MPITASKS_PER_NODE>64</MAX_MPITASKS_PER_NODE>
      <PROJECT_REQUIRED>TRUE</PROJECT_REQUIRED>

      <mpirun mpilib="openmpi">
        <executable>srun</executable>
      </mpirun>
      <module_system type="module" allow_error="true">
        <init_path lang="perl">/cvmfs/soft.computecanada.ca/custom/software/lmod/lmod/init/perl</init_path>
        <init_path lang="python">/cvmfs/soft.computecanada.ca/custom/software/lmod/lmod/init/env_modules_python.py</init_path>
        <init_path lang="csh">/cvmfs/soft.computecanada.ca/custom/software/lmod/lmod/init/csh</init_path>
        <init_path lang="sh">/cvmfs/soft.computecanada.ca/custom/software/lmod/lmod/init/sh</init_path>
        <cmd_path lang="perl">/cvmfs/soft.computecanada.ca/custom/software/lmod/lmod/libexec/lmod perl</cmd_path>
        <cmd_path lang="python">/cvmfs/soft.computecanada.ca/custom/software/lmod/lmod/libexec/lmod python</cmd_path>
        <cmd_path lang="csh">module</cmd_path>
        <cmd_path lang="sh">module</cmd_path>
        <modules>
        <command name="purge"/>
      <command name="load">StdEnv/2023</command>
        </modules>
        <modules compiler="intel">
      <command name="load">intel/2023.2.1</command>
      <command name="load">git-annex/10.20231129</command>
      <command name="load">cmake/3.27.7</command>
        </modules>
        <modules mpilib="openmpi">
          <command name="load">openmpi/4.1.5</command>
          <command name="load">hdf5-mpi/1.14.2</command>
          <command name="load">netcdf-c++4-mpi/4.3.1</command>
          <command name="load">netcdf-fortran-mpi/4.6.1</command>
          <command name="load">netcdf-mpi/4.9.2</command>
      <command name="load">xml-libxml/2.0208</command>
      <command name="load">flexiblas/3.3.1</command>
        </modules>
      </module_system>
      <environment_variables>
              <env name="NETCDF_PATH">/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v4/MPI/intel2023/openmpi4/pnetcdf/1.12.3</env>
              <env name="NETCDF_FORTRAN_PATH">/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v4/MPI/intel2023/openmpi4/netcdf-fortran-mpi/4.6.1/</env>
              <env name="NETCDF_C_PATH">/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v4/MPI/intel2023/openmpi4/netcdf-c++4-mpi/4.3.1/</env>
              <env name="NETLIB_LAPACK_PATH">/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Core/imkl/2023.2.0/mkl/2023.2.0/</env>
          <env name="OMP_STACKSIZE">256M</env>
              <env name="I_MPI_CC">icc</env>
              <env name="I_MPI_FC">ifort</env>
              <env name="I_MPI_F77">ifort</env>
              <env name="I_MPI_F90">ifort</env>
              <env name="I_MPI_CXX">icpc</env>
      </environment_variables>
      <resource_limits>
        <resource name="RLIMIT_STACK">300000000</resource>
      </resource_limits>
    </machine>
  </config_machines>
  }}`{=mediawiki} `</tab>`{=html}

  `<tab name="Niagara">`{=html} `{{File
    |name=~/.cime/config_machines.xml
    |lang="xml"
    |contents=
  <?xml version="1.0"?>

  <config_machines version="2.0">
    <machine MACH="niagara">
      <DESC>https://docs.alliancecan.ca/wiki/Niagara</DESC>
      <NODENAME_REGEX>nia.*.scinet.local</NODENAME_REGEX>

      <OS>LINUX</OS>
      <COMPILERS>intel,gnu</COMPILERS>
      <MPILIBS>openmpi</MPILIBS>

      <PROJECT>def-EDIT_THIS</PROJECT>
      <CHARGE_ACCOUNT>def-EDIT_THIS</CHARGE_ACCOUNT>

      <CIME_OUTPUT_ROOT>/scratch/$USER/cesm/output</CIME_OUTPUT_ROOT>
      <DIN_LOC_ROOT>/scratch/$USER/cesm/inputdata</DIN_LOC_ROOT>
      <DIN_LOC_ROOT_CLMFORC>${DIN_LOC_ROOT}/atm/datm7</DIN_LOC_ROOT_CLMFORC>
      <DOUT_S_ROOT>$CIME_OUTPUT_ROOT/archive/case</DOUT_S_ROOT>
      <GMAKE>make</GMAKE>
      <GMAKE_J>8</GMAKE_J>
      <BATCH_SYSTEM>slurm</BATCH_SYSTEM>
      <SUPPORTED_BY>support@tech.alliancecan.ca</SUPPORTED_BY>
      <MAX_TASKS_PER_NODE>40</MAX_TASKS_PER_NODE>
      <MAX_MPITASKS_PER_NODE>40</MAX_MPITASKS_PER_NODE>
      <PROJECT_REQUIRED>TRUE</PROJECT_REQUIRED>

      <mpirun mpilib="openmpi">
        <executable>srun</executable>
      </mpirun>
      <module_system type="module" allow_error="true">
        <init_path lang="perl">/cvmfs/soft.computecanada.ca/custom/software/lmod/lmod/init/perl</init_path>
        <init_path lang="python">/cvmfs/soft.computecanada.ca/custom/software/lmod/lmod/init/env_modules_python.py</init_path>
        <init_path lang="csh">/cvmfs/soft.computecanada.ca/custom/software/lmod/lmod/init/csh</init_path>
        <init_path lang="sh">/cvmfs/soft.computecanada.ca/custom/software/lmod/lmod/init/sh</init_path>
        <cmd_path lang="perl">/cvmfs/soft.computecanada.ca/custom/software/lmod/lmod/libexec/lmod perl</cmd_path>
        <cmd_path lang="python">/cvmfs/soft.computecanada.ca/custom/software/lmod/lmod/libexec/lmod python</cmd_path>
        <cmd_path lang="csh">module</cmd_path>
        <cmd_path lang="sh">module</cmd_path>
        <modules>
        <command name="purge"/>
      <command name="load">StdEnv/2023</command>
        </modules>
        <modules compiler="intel">
      <command name="load">intel/2023.2.1</command>
      <command name="load">git-annex/10.20231129</command>
      <command name="load">cmake/3.27.7</command>
        </modules>
        <modules mpilib="openmpi">
          <command name="load">openmpi/4.1.5</command>
          <command name="load">hdf5-mpi/1.14.2</command>
          <command name="load">netcdf-c++4-mpi/4.3.1</command>
          <command name="load">netcdf-fortran-mpi/4.6.1</command>
          <command name="load">netcdf-mpi/4.9.2</command>
      <command name="load">xml-libxml/2.0208</command>
      <command name="load">flexiblas/3.3.1</command>
        </modules>
      </module_system>
      <environment_variables>
              <env name="NETCDF_PATH">/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v4/MPI/intel2023/openmpi4/pnetcdf/1.12.3</env>
              <env name="NETCDF_FORTRAN_PATH">/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v4/MPI/intel2023/openmpi4/netcdf-fortran-mpi/4.6.1/</env>
              <env name="NETCDF_C_PATH">/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v4/MPI/intel2023/openmpi4/netcdf-c++4-mpi/4.3.1/</env>
              <env name="NETLIB_LAPACK_PATH">/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Core/imkl/2023.2.0/mkl/2023.2.0/</env>
          <env name="OMP_STACKSIZE">256M</env>
              <env name="I_MPI_CC">icc</env>
              <env name="I_MPI_FC">ifort</env>
              <env name="I_MPI_F77">ifort</env>
              <env name="I_MPI_F90">ifort</env>
              <env name="I_MPI_CXX">icpc</env>
      </environment_variables>
      <resource_limits>
        <resource name="RLIMIT_STACK">300000000</resource>
      </resource_limits>
    </machine>
  </config_machines>
  }}`{=mediawiki} `</tab>`{=html}

  `<tab name="Rorqual">`{=html} Note: despite the Intel software dependencies, the below configuration works on Rorqual\'s AMD processors. `{{File
    |name=~/.cime/config_machines.xml
    |lang="xml"
    |contents=
  <?xml version="1.0"?>

  <config_machines version="2.0">
    <machine MACH="rorqual">
      <DESC>https://docs.alliancecan.ca/wiki/Rorqual/en</DESC>
      <NODENAME_REGEX>r.*\.rorqual\.calcul\.quebec</NODENAME_REGEX>

      <OS>LINUX</OS>
      <COMPILERS>intel,gnu</COMPILERS>
      <MPILIBS>openmpi</MPILIBS>

      <PROJECT>def-EDIT_THIS</PROJECT>
      <CHARGE_ACCOUNT>def-EDIT_THIS</CHARGE_ACCOUNT>

      <CIME_OUTPUT_ROOT>/scratch/$USER/cesm/output</CIME_OUTPUT_ROOT>
      <DIN_LOC_ROOT>/scratch/$USER/cesm/inputdata</DIN_LOC_ROOT>
      <DIN_LOC_ROOT_CLMFORC>${DIN_LOC_ROOT}/atm/datm7</DIN_LOC_ROOT_CLMFORC>
      <DOUT_S_ROOT>$CIME_OUTPUT_ROOT/archive/case</DOUT_S_ROOT>
      <GMAKE>make</GMAKE>
      <GMAKE_J>8</GMAKE_J>
      <BATCH_SYSTEM>slurm</BATCH_SYSTEM>
      <SUPPORTED_BY>support@tech.alliancecan.ca</SUPPORTED_BY>
      <MAX_TASKS_PER_NODE>192</MAX_TASKS_PER_NODE>
      <MAX_MPITASKS_PER_NODE>192</MAX_MPITASKS_PER_NODE>
      <PROJECT_REQUIRED>TRUE</PROJECT_REQUIRED>

      <mpirun mpilib="openmpi">
        <executable>srun</executable>
      </mpirun>
      <module_system type="module" allow_error="true">
        <init_path lang="perl">/cvmfs/soft.computecanada.ca/custom/software/lmod/lmod/init/perl</init_path>
        <init_path lang="python">/cvmfs/soft.computecanada.ca/custom/software/lmod/lmod/init/env_modules_python.py</init_path>
        <init_path lang="csh">/cvmfs/soft.computecanada.ca/custom/software/lmod/lmod/init/csh</init_path>
        <init_path lang="sh">/cvmfs/soft.computecanada.ca/custom/software/lmod/lmod/init/sh</init_path>
        <cmd_path lang="perl">/cvmfs/soft.computecanada.ca/custom/software/lmod/lmod/libexec/lmod perl</cmd_path>
        <cmd_path lang="python">/cvmfs/soft.computecanada.ca/custom/software/lmod/lmod/libexec/lmod python</cmd_path>
        <cmd_path lang="csh">module</cmd_path>
        <cmd_path lang="sh">module</cmd_path>
        <modules>
        <command name="purge"/>
      <command name="load">StdEnv/2023</command>
        </modules>
        <modules compiler="intel">
      <command name="load">intel/2023.2.1</command>
      <command name="load">git-annex/10.20231129</command>
      <command name="load">cmake/3.27.7</command>
        </modules>
        <modules mpilib="openmpi">
          <command name="load">openmpi/4.1.5</command>
          <command name="load">hdf5-mpi/1.14.2</command>
          <command name="load">netcdf-c++4-mpi/4.3.1</command>
          <command name="load">netcdf-fortran-mpi/4.6.1</command>
          <command name="load">netcdf-mpi/4.9.2</command>
      <command name="load">xml-libxml/2.0208</command>
      <command name="load">flexiblas/3.3.1</command>
        </modules>
      </module_system>
      <environment_variables>
              <env name="NETCDF_PATH">/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v4/MPI/intel2023/openmpi4/pnetcdf/1.12.3</env>
              <env name="NETCDF_FORTRAN_PATH">/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v4/MPI/intel2023/openmpi4/netcdf-fortran-mpi/4.6.1/</env>
              <env name="NETCDF_C_PATH">/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v4/MPI/intel2023/openmpi4/netcdf-c++4-mpi/4.3.1/</env>
              <env name="NETLIB_LAPACK_PATH">/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Core/imkl/2023.2.0/mkl/2023.2.0/</env>
          <env name="OMP_STACKSIZE">256M</env>
              <env name="I_MPI_CC">icc</env>
              <env name="I_MPI_FC">ifort</env>
              <env name="I_MPI_F77">ifort</env>
              <env name="I_MPI_F90">ifort</env>
              <env name="I_MPI_CXX">icpc</env>
      </environment_variables>
      <resource_limits>
        <resource name="RLIMIT_STACK">300000000</resource>
      </resource_limits>
    </machine>
  </config_machines>
  }}`{=mediawiki} `</tab>`{=html} `</tabs>`{=html}
- Validate your XML machine file with the following commands:
- Check the official template for additional parameters:

## Local batch file {#local_batch_file}

- Create and edit the file `~/.cime/config_batch.xml` from the following minimal content:
  `<tabs>`{=html} `<tab name="Beluga">`{=html} `</nowiki>`{=html}`</directive>`{=html}

  `     ``<directive>`{=html}`--nodes=``</directive>`{=html}\
  `     ``<directive>`{=html}`--ntasks-per-node=``</directive>`{=html}\
  `     ``<directive>`{=html}`--output=``</directive>`{=html}\
  `     ``<directive>`{=html}`--exclusive``</directive>`{=html}\
  `     ``<directive>`{=html}`--mem=0``</directive>`{=html}\
  `   ``</directives>`{=html}\
  `   ``<unknown_queue_directives>`{=html}`regular``</unknown_queue_directives>`{=html}\
  `   ``<queues>`{=html}\
  `     ``<queue walltimemax="12:00:00" nodemin="1" nodemax="576">`{=html}`beluga``</queue>`{=html}\
  `   ``</queues>`{=html}\
  ` ``</batch_system>`{=html}

  }} `</tab>`{=html}

  `<tab name="Cedar">`{=html} `</nowiki>`{=html}`</directive>`{=html}

  `     ``<directive>`{=html}`--nodes=``</directive>`{=html}\
  `     ``<directive>`{=html}`--ntasks-per-node=``</directive>`{=html}\
  `     ``<directive>`{=html}`--output=``</directive>`{=html}\
  `     ``<directive>`{=html}`--exclusive``</directive>`{=html}\
  `     ``<directive>`{=html}`--mem=0``</directive>`{=html}\
  `   ``</directives>`{=html}\
  `    `\
  `  ``<unknown_queue_directives>`{=html}`regular``</unknown_queue_directives>`{=html}\
  `   ``<queues>`{=html}\
  `    ``<queue walltimemax="12:00:00" nodemin="1" nodemax="1360">`{=html}`cedar``</queue>`{=html}\
  `   ``</queues>`{=html}\
  ` ``</batch_system>`{=html}

  `</file>`{=html} }} `</tab>`{=html}

  `<tab name="Graham">`{=html} `</nowiki>`{=html}`</directive>`{=html}

  `     ``<directive>`{=html}`--nodes=``</directive>`{=html}\
  `     ``<directive>`{=html}`--ntasks-per-node=``</directive>`{=html}\
  `     ``<directive>`{=html}`--output=``</directive>`{=html}\
  `     ``<directive>`{=html}`--exclusive``</directive>`{=html}\
  `     ``<directive>`{=html}`--mem=0``</directive>`{=html}\
  `   ``</directives>`{=html}\
  `    `\
  `  ``<unknown_queue_directives>`{=html}`regular``</unknown_queue_directives>`{=html}\
  `   ``<queues>`{=html}\
  `    ``<queue walltimemax="12:00:00" nodemin="1" nodemax="136">`{=html}`graham``</queue>`{=html}\
  `   ``</queues>`{=html}\
  ` ``</batch_system>`{=html}

  `</file>`{=html} }} `</tab>`{=html}

  `<tab name="Narval">`{=html} `</nowiki>`{=html}`</directive>`{=html}

  `     ``<directive>`{=html}`--nodes=``</directive>`{=html}\
  `     ``<directive>`{=html}`--ntasks-per-node=``</directive>`{=html}\
  `     ``<directive>`{=html}`--output=``</directive>`{=html}\
  `     ``<directive>`{=html}`--exclusive``</directive>`{=html}\
  `     ``<directive>`{=html}`--mem=0``</directive>`{=html}\
  `   ``</directives>`{=html}\
  `   ``<unknown_queue_directives>`{=html}`regular``</unknown_queue_directives>`{=html}\
  `   ``<queues>`{=html}\
  `     ``<queue walltimemax="12:00:00" nodemin="1" nodemax="1145">`{=html}`narval``</queue>`{=html}\
  `   ``</queues>`{=html}\
  ` ``</batch_system>`{=html}

  }} `</tab>`{=html}

  `<tab name="Niagara">`{=html} `</nowiki>`{=html}`</directive>`{=html}

  `     ``<directive>`{=html}`--nodes=``</directive>`{=html}\
  `     ``<directive>`{=html}`--ntasks-per-node=``</directive>`{=html}\
  `     ``<directive>`{=html}`--output=``</directive>`{=html}\
  `     ``<directive>`{=html}`--exclusive``</directive>`{=html}\
  `     ``<directive>`{=html}`--mem=0``</directive>`{=html}\
  `     ``<directive>`{=html}`--constraint=[skylakecascade]``</directive>`{=html}\
  `   ``</directives>`{=html}\
  `    `\
  `  ``<unknown_queue_directives>`{=html}`regular``</unknown_queue_directives>`{=html}\
  `   ``<queues>`{=html}\
  `    ``<queue walltimemax="12:00:00" nodemin="1" nodemax="2024">`{=html}`niagara``</queue>`{=html}\
  `   ``</queues>`{=html}\
  ` ``</batch_system>`{=html}

  `</file>`{=html} }} `</tab>`{=html}

  `<tab name="Rorqual">`{=html} `</nowiki>`{=html}`</directive>`{=html}

  `     ``<directive>`{=html}`--nodes=``</directive>`{=html}\
  `     ``<directive>`{=html}`--ntasks-per-node=``</directive>`{=html}\
  `     ``<directive>`{=html}`--output=``</directive>`{=html}\
  `     ``<directive>`{=html}`--exclusive``</directive>`{=html}\
  `     ``<directive>`{=html}`--mem=0``</directive>`{=html}\
  `   ``</directives>`{=html}\
  `   `\
  `   ``<unknown_queue_directives>`{=html}`regular``</unknown_queue_directives>`{=html}\
  `   ``<queues>`{=html}\
  `     ``<queue walltimemax="12:00:00" nodemin="1" nodemax="670">`{=html}`rorqual``</queue>`{=html}\
  `   ``</queues>`{=html}\
  ` ``</batch_system>`{=html}

  }} `</tab>`{=html} `</tabs>`{=html}
- Validate your XML batch file with the following commands:
- Check the documentation for additional `<b>`{=html}[configuration parameters and examples](https://esmci.github.io/cime/versions/maint-5.6/html/xml_files/cesm.html#cimeroot-config-cesm-machines)`</b>`{=html}.

## Local compilers file {#local_compilers_file}

- Create and edit the file `~/.cime/config_compilers.xml` from the following minimal content per cluster:
  `<tabs>`{=html} `<tab name="Beluga">`{=html} `{{File
    |name=~/.cime/config_compilers.xml
    |lang="xml"
    |contents=
  <?xml version="1.0"?>

      <compiler MACH="beluga">
        <CPPDEFS>
          <!-- these flags enable nano timers -->
          <append MODEL="gptl"> -DHAVE_NANOTIME -DBIT64 -DHAVE_VPRINTF -DHAVE_BACKTRACE -DHAVE_SLASHPROC -DHAVE_COMM_F2C -DHAVE_TIMES -DHAVE_GETTIMEOFDAY </append>
        </CPPDEFS>
        <NETCDF_PATH>$ENV{NETCDF_FORTRAN_ROOT}</NETCDF_PATH>
        <PIO_FILESYSTEM_HINTS>lustre</PIO_FILESYSTEM_HINTS>
        <PNETCDF_PATH>$ENV{PARALLEL_NETCDF_ROOT}</PNETCDF_PATH>
        <SLIBS>
          <append>-L$(NETCDF_PATH)/lib -lnetcdff -L$(NETCDF_C_ROOT)/lib -lnetcdf -L$(NETLIB_LAPACK_PATH)/lib/intel64 -lmkl -ldl </append>
        </SLIBS>
      </compiler>
  }}`{=mediawiki} `</tab>`{=html}

  `<tab name="Cedar">`{=html} `{{File
    |name=~/.cime/config_compilers.xml
    |lang="xml"
    |contents=
  <?xml version="1.0"?>

      <compiler MACH="cedar">
        <CPPDEFS>
          <!-- these flags enable nano timers -->
          <append MODEL="gptl"> -DHAVE_NANOTIME -DBIT64 -DHAVE_VPRINTF -DHAVE_BACKTRACE -DHAVE_SLASHPROC -DHAVE_COMM_F2C -DHAVE_TIMES -DHAVE_GETTIMEOFDAY </append>
        </CPPDEFS>
        <NETCDF_PATH>$ENV{NETCDF_FORTRAN_ROOT}</NETCDF_PATH>
        <PIO_FILESYSTEM_HINTS>lustre</PIO_FILESYSTEM_HINTS>
        <PNETCDF_PATH>$ENV{PARALLEL_NETCDF_ROOT}</PNETCDF_PATH>
        <SLIBS>
          <append>-L$(NETCDF_PATH)/lib -lnetcdff -L$(NETCDF_C_ROOT)/lib -lnetcdf -L$(NETLIB_LAPACK_PATH)/lib/intel64 -lmkl -ldl </append>
        </SLIBS>
      </compiler>
  }}`{=mediawiki} `</tab>`{=html}

  `<tab name="Graham">`{=html} `{{File
    |name=~/.cime/config_compilers.xml
    |lang="xml"
    |contents=
  <?xml version="1.0"?>

      <compiler MACH="graham">
        <CPPDEFS>
          <!-- these flags enable nano timers -->
          <append MODEL="gptl"> -DHAVE_NANOTIME -DBIT64 -DHAVE_VPRINTF -DHAVE_BACKTRACE -DHAVE_SLASHPROC -DHAVE_COMM_F2C -DHAVE_TIMES -DHAVE_GETTIMEOFDAY </append>
        </CPPDEFS>
        <NETCDF_PATH>$ENV{NETCDF_FORTRAN_ROOT}</NETCDF_PATH>
        <PIO_FILESYSTEM_HINTS>lustre</PIO_FILESYSTEM_HINTS>
        <PNETCDF_PATH>$ENV{PARALLEL_NETCDF_ROOT}</PNETCDF_PATH>
        <SLIBS>
          <append>-L$(NETCDF_PATH)/lib -lnetcdff -L$(NETCDF_C_ROOT)/lib -lnetcdf -L$(NETLIB_LAPACK_PATH)/lib/intel64 -lmkl -ldl </append>
        </SLIBS>
      </compiler>
  }}`{=mediawiki} `</tab>`{=html}

  `<tab name="Narval">`{=html} `{{File
    |name=~/.cime/config_compilers.xml
    |lang="xml"
    |contents=
  <?xml version="1.0"?>

      <compiler MACH="narval">
        <CPPDEFS>
          <!-- these flags enable nano timers -->
          <append MODEL="gptl"> -DHAVE_NANOTIME -DBIT64 -DHAVE_VPRINTF -DHAVE_BACKTRACE -DHAVE_SLASHPROC -DHAVE_COMM_F2C -DHAVE_TIMES -DHAVE_GETTIMEOFDAY </append>
        </CPPDEFS>
        <NETCDF_PATH>$ENV{NETCDF_FORTRAN_ROOT}</NETCDF_PATH>
        <PIO_FILESYSTEM_HINTS>lustre</PIO_FILESYSTEM_HINTS>
        <PNETCDF_PATH>$ENV{PARALLEL_NETCDF_ROOT}</PNETCDF_PATH>
        <SLIBS>
          <append>-L$(NETCDF_PATH)/lib -lnetcdff -L$(NETCDF_C_ROOT)/lib -lnetcdf -L$(NETLIB_LAPACK_PATH)/lib/intel64 -lmkl -ldl </append>
        </SLIBS>
      </compiler>
  }}`{=mediawiki} `</tab>`{=html}

  `<tab name="Niagara">`{=html} `{{File
    |name=~/.cime/config_compilers.xml
    |lang="xml"
    |contents=
  <?xml version="1.0"?>

      <compiler MACH="niagara">
        <CPPDEFS>
          <!-- these flags enable nano timers -->
          <append MODEL="gptl"> -DHAVE_NANOTIME -DBIT64 -DHAVE_VPRINTF -DHAVE_BACKTRACE -DHAVE_SLASHPROC -DHAVE_COMM_F2C -DHAVE_TIMES -DHAVE_GETTIMEOFDAY </append>
        </CPPDEFS>
        <NETCDF_PATH>$ENV{NETCDF_FORTRAN_ROOT}</NETCDF_PATH>
        <PIO_FILESYSTEM_HINTS>lustre</PIO_FILESYSTEM_HINTS>
        <PNETCDF_PATH>$ENV{PARALLEL_NETCDF_ROOT}</PNETCDF_PATH>
        <SLIBS>
          <append>-L$(NETCDF_PATH)/lib -lnetcdff -L$(NETCDF_C_ROOT)/lib -lnetcdf -L$(NETLIB_LAPACK_PATH)/lib/intel64 -lmkl -ldl </append>
        </SLIBS>
      </compiler>
  }}`{=mediawiki} `</tab>`{=html}

  `<tab name="Rorqual">`{=html} `{{File
    |name=~/.cime/config_compilers.xml
    |lang="xml"
    |contents=
  <?xml version="1.0"?>

      <compiler MACH="rorqual">
        <CPPDEFS>
          <append MODEL="gptl"> -DHAVE_NANOTIME -DBIT64 -DHAVE_VPRINTF -DHAVE_BACKTRACE -DHAVE_SLASHPROC -DHAVE_COMM_F2C -DHAVE_TIMES -DHAVE_GETTIMEOFDAY </append>
        </CPPDEFS>
        <NETCDF_PATH>$ENV{NETCDF_FORTRAN_ROOT}</NETCDF_PATH>
        <PIO_FILESYSTEM_HINTS>lustre</PIO_FILESYSTEM_HINTS>
        <PNETCDF_PATH>$ENV{PARALLEL_NETCDF_ROOT}</PNETCDF_PATH>
        <SLIBS>
          <append>-L$(NETCDF_PATH)/lib -lnetcdff -L$(NETCDF_C_ROOT)/lib -lnetcdf -L$(NETLIB_LAPACK_PATH)/lib/intel64 -lmkl -ldl </append>
        </SLIBS>
      </compiler>
  }}`{=mediawiki} `</tab>`{=html} `</tabs>`{=html}
- Validate your XML compiler file with the following commands:

## Creating a test case {#creating_a_test_case}

The following commands assume the default model `cesm` and the `current` machine:

# Reference

- [Main website](https://www.cesm.ucar.edu/)
  - [CESM Quickstart Guide (CESM2.1)](https://escomp.github.io/CESM/versions/cesm2.1/html/)
  - [CESM Coupled Model XML Files](https://esmci.github.io/cime/versions/maint-5.6/html/xml_files/cesm.html)
