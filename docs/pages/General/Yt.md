---
title: "Yt/en"
url: "https://docs.alliancecan.ca/wiki/Yt/en"
category: "General"
last_modified: "2019-12-16T15:46:11Z"
page_id: 10595
display_title: "Yt"
---

= YT rendering on clusters =

To install YT for CPU rendering on a cluster in your own directory, please do

 $ module load python mpi4py
 $ virtualenv astro    # install Python tools in your $HOME/astro
 $ source ~/astro/bin/activate
 $ pip install cython
 $ pip install numpy
 $ pip install yt

Then, in normal use, simply load the environment and start python

 $ source ~/astro/bin/activate   # load the environment
 $ python
 ...
 $ deactivate

We assume that you have downloaded the sample dataset Enzo_64 from http://yt-project.org/data. Start with the following script `grids.py` to render 90 frames rotating the dataset around the vertical axis

 import yt
 from numpy import pi
 yt.enable_parallelism()   # turn on MPI parallelism via mpi4py
 ds = yt.load("Enzo_64/DD0043/data0043")
 sc = yt.create_scene(ds, ('gas', 'density'))
 cam = sc.camera
 cam.resolution = (1024, 1024)   # resolution of each frame
 sc.annotate_domain(ds, color=[1, 1, 1, 0.005])   # draw the domain boundary [r,g,b,alpha]
 sc.annotate_grids(ds, alpha=0.005)   # draw the grid boundaries
 sc.save('frame0000.png', sigma_clip=4)
 nspin = 90
 for i in cam.iter_rotate(pi, nspin):   # rotate by 180 degrees over nspin frames
     sc.save('frame%04d.png' % (i+1), sigma_clip=4)

and the job submission script `yt-mpi.sh`

 #!/bin/bash
 #SBATCH --time=0:30:00   # walltime in d-hh:mm or hh:mm:ss format
 #SBATCH --ntasks=4       # number of MPI processes
 #SBATCH --mem-per-cpu=3800
 #SBATCH --account=...
 source $HOME/astro/bin/activate
 srun python grids.py

Then submit the job with `sbatch yt-mpi.sh`, wait for it to finish, and then create a movie at 30fps

 $ ffmpeg -r 30 -i frame%04d.png -c:v libx264 -pix_fmt yuv420p -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" grids.mp4