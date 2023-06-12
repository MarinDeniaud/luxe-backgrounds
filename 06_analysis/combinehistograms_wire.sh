# environment - bdsim, root, geant4 etc
source /cvmfs/beam-physics.cern.ch/bdsim/x86_64-centos7-gcc11-opt/bdsim-env-v1.7.0-rc-g4v10.7.2.3-ftfp-boost.sh

# combine command
while read tag; do rebdsimCombine "$tag"_merged_hist.root *"$tag"_hist.root; done < tagfilelistwire
