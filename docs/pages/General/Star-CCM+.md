---
title: "Star-CCM+"
url: "https://docs.alliancecan.ca/wiki/Star-CCM%2B"
category: "General"
last_modified: "2026-01-02T06:33:23Z"
page_id: 4355
display_title: "Star-CCM+"
---

STAR-CCM+ is a multidisciplinary engineering simulation suite to model acoustics, fluid dynamics, heat transfer, rheology, multiphase flows, particle flows, solid mechanics, reacting flows, electrochemistry, and electromagnetics. It is developed by Siemens.

= License limitations =
We have the authorization to host STAR-CCM+ binaries on our servers, but we don't provide licenses. You will need to have your own license in order to use this software. A remote POD license can be purchased directly from Siemens. Alternatively, a local license hosted at your institution can be used, providing it can be accessed through the firewall from the cluster where jobs are to be run.

== Configuring your account ==
To configure your account to use a license server with the Star-CCM+ module, create a license file $HOME/.licenses/starccm.lic with the following layout:

where  and  should be changed to specify the hostname (or ip address) and the static vendor port of the license server respectively.  Note that manually setting CDLMD_LICENSE_FILE equal to @ in your slurm script is not required;  Instead, when a Star-CCM+ module is loaded this variable is automatically set to your $HOME/.licenses/starccm.lic file.

=== POD license file ===

Researchers with a POD license purchased from Siemens must manually set the LM_PROJECT environment variable equal to YOUR CD-ADAPCO PROJECT ID in your slurm script.  Also the ~/.licenses/starccm.lic file should be configured as follows on each cluster:

= Cluster batch job submission =

When submitting jobs on a cluster for the first time, you must set up the environment to use your license. If you are using Siemans remote pay-on-usage license server then create a ~/.licenses/starccm.lic file as shown in the Configuring your account- POD license file section above and license checkouts should immediately work.  If however you are using an institutional license server, then after creating your ~/.licenses/starccm.lic file you must also submit a problem ticket to technical support so we can help co-ordinate the necessary one time network firewall changes required to access it (assuming the server has never been setup to be accessed from the Alliance cluster you will be using).   If you still have problems getting the licensing to work then try removing or renaming file ~/.flexlmrc since previous search paths and/or license server settings maybe stored in it.  Note that temporary output files from starccm jobs runs may accumulate in hidden directories named ~/.star-version_number consuming valuable quota space.  These can be removed by periodically running rm -ri ~/.starccm* and replying yes when prompted.

== Slurm scripts ==

 awk '{print $2}')
port=$(cat $CDLMD_LICENSE_FILE  grep -Eo '[0-9]+$')
nmap $server -Pn -p $port  grep -v '^$'; echo

export FLEXIBLAS=StarMKL
echo "FLEXIBLAS=$FLEXIBLAS"
STAR_MPI="-mpi intel"
STAR_FABRIC="-fabric tcp"

if [ -n "$LM_PROJECT" ]; then
   echo "Siemens PoD license server ..."
   starccm+ -jvmargs -Xmx4G -jvmargs -Djava.io.tmpdir=$SLURM_TMPDIR -batch -power -podkey $LM_PROJECT -np $NCORE -nbuserdir $SLURM_TMPDIR -machinefile $SLURM_TMPDIR/machinefile $JAVA_FILE $SIM_FILE $STAR_MPI $STAR_FABRIC
else
   echo "Institutional license server ..."
   [ $(command -v lmutil) ] && lmutil lmstat -c ~/.licenses/starccm.lic -a  egrep "license1UPuse$USER"; echo
   starccm+ -jvmargs -Xmx4G -jvmargs -Djava.io.tmpdir=$SLURM_TMPDIR -batch -np $NCORE -nbuserdir $SLURM_TMPDIR -machinefile $SLURM_TMPDIR/machinefile $JAVA_FILE $SIM_FILE $STAR_MPI $STAR_FABRIC
fi
}}

 awk '{print $2}')
port=$(cat $CDLMD_LICENSE_FILE  grep -Eo '[0-9]+$')
nmap $server -Pn -p $port  grep -v '^$'; echo

export FLEXIBLAS=NETLIB
STAR_MPI="-mpi openmpi"
if [ "$RSNT_CPU_VENDOR_ID" == intel ]; then
  export FLEXIBLAS=StarMKL
  STAR_MPI="-mpi intel"
elif [ "$RSNT_CPU_VENDOR_ID" == amd ]; then
  export FLEXIBLAS=StarAOCL
fi
echo "FLEXIBLAS=$FLEXIBLAS"

if [ "${EBVERSIONSTARCCM:0:2}" -lt 20 ]; then
  STAR_FABRIC="-fabric ofi"
else
  STAR_FABRIC="-fabric ucx"
fi

if [ -n "$LM_PROJECT" ]; then
   echo "Siemens PoD license server ..."
   starccm+ -jvmargs -Xmx4G -jvmargs -Djava.io.tmpdir=$SLURM_TMPDIR -batch -power -podkey $LM_PROJECT -np $NCORE -nbuserdir $SLURM_TMPDIR -machinefile $SLURM_TMPDIR/machinefile $JAVA_FILE $SIM_FILE $STAR_MPI $STAR_FABRIC
else
   echo "Institutional license server ..."
   [ $(command -v lmutil) ] && lmutil lmstat -c ~/.licenses/starccm.lic -a  egrep "license1UPuse$USER"; echo
   starccm+ -jvmargs -Xmx4G -jvmargs -Djava.io.tmpdir=$SLURM_TMPDIR -batch -np $NCORE -nbuserdir $SLURM_TMPDIR -machinefile $SLURM_TMPDIR/machinefile $JAVA_FILE $SIM_FILE $STAR_MPI $STAR_FABRIC
fi
}}

 sleep 5
          echo "Attempt number: "$I
          if [ -n "$LM_PROJECT" ]; then
          echo "Siemens PoD license server ..."
          starccm+ -jvmargs -Xmx4G -jvmargs -Djava.io.tmpdir=$SLURM_TMPDIR -batch -power -podkey $LM_PROJECT -np $NCORE -nbuserdir $SLURM_TMPDIR -machinefile $SLURM_TMPDIR/machinefile $JAVA_FILE $SIM_FILE $STAR_MPI
        else
          echo "Institutional license server ..."
          starccm+ -jvmargs -Xmx4G -jvmargs -Djava.io.tmpdir=$SLURM_TMPDIR -batch -np $NCORE -nbuserdir $SLURM_TMPDIR -machinefile $SLURM_TMPDIR/machinefile $JAVA_FILE $SIM_FILE $STAR_MPI
        fi
        RET=$?
        i=$((i+1))
done
exit $RET
}}

= Graphical use =

To run starccm+ in graphical mode it is recommended to use an  OnDemand or JupyterLab system to start a remote desktop.  In addition to configuring ~/.licenses/starccm.lic research groups with a POD license should also run export LM_PROJECT='CD-ADAPCO PROJECT ID'and optionally append -power to the list of command starccm+ line options below.  Note that module avail starccm-mixed will display which starccm versions are available within the StdEnv/version that you currently have loaded.  Alternatively running module spider starccm-mixed will show all available starccm module versions available within all StdEnv module versions.

== OnDemand ==
1. Connect to an OnDemand system using one of the following URLs in your laptop browser :
 NIBI: https://ondemand.sharcnet.ca
 FIR: https://jupyterhub.fir.alliancecan.ca
 RORQUAL: https://jupyterhub.rorqual.alliancecan.ca
 TRILLIUM: https://ondemand.scinet.utoronto.ca
2. Open a new terminal window in your desktop and run one of:
: STAR-CCM+ 18.04.008 (or newer versions)
:: module load StdEnv/2023  (default)
:: module load starccm-mixed/20.04.007 **OR** starccm/20.04.007-R8
:: starccm+ -rr server
: STAR-CCM+ 15.04.010 --> 18.02.008 (version range)
:: module load StdEnv/2020 (unsupported)
:: module load starccm-mixed/15.04.010 **OR** starccm/15.04.010-R8
:: starccm+ -mesa

== VncViewer ==

1. Connect with a VncViewer client to a login or compute node by following TigerVNC
2. Open a new terminal window in your desktop and run one of:
: STAR-CCM+ 18.04.008 (or newer versions)
:: module load StdEnv/2023  (default)
:: module load starccm-mixed/20.04.007 **OR** starccm/20.04.007-R8
:: starccm+ -rr server
: STAR-CCM+ 15.04.010 --> 18.02.008 (version range)
:: module load StdEnv/2020 (unsupported)
:: module load starccm-mixed/17.02.007 **OR** starccm/17.02.007-R8
::  starccm+