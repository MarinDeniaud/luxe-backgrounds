# environment - bdsim, root, geant4 etc                                                                                               
source /cvmfs/beam-physics.cern.ch/bdsim/x86_64-centos7-gcc11-opt/bdsim-env-develop-g4v11.0.2-boost.sh

# combine command
while read tag; do rebdsimCombine "$tag"_merged_hist.root *"$tag"_hist.root; done < tagfilelist
