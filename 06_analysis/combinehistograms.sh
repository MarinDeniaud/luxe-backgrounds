# environment - bdsim, root, geant4 etc                                                                                               
source /cvmfs/beam-physics.cern.ch/bdsim/x86_64-centos7-gcc8-opt/bdsim-env-v1.6.0-g4v10.7.2.3.sh

# combine command
while read tag; do rebdsimCombine "$tag"_merged_hist.root *"$tag"_hist.root; done < tagfilelist
