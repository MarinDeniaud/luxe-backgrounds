#!/bin/sh

# environment - bdsim, root, geant4 etc
source /cvmfs/beam-physics.cern.ch/bdsim/x86_64-centos7-gcc11-opt/bdsim-env-develop-g4v11.0.2-boost.sh

# echo inputs
INPUTFILE=$1
OUTPUTFILE=$2
SEEDVALUE=$(($3+$4))
OFFSET=$5
NGEN=$6
MODELDIR=$7
echo "input file    = "${INPUTFILE}
echo "output file   = "${OUTPUTFILE}
echo "seed value    = "${SEEDVALUE}
echo "wire offset   = "${OFFSET}
echo "ngenerate     = "${NGEN}
echo "model dir     = "${MODELDIR}

date
echo ""
hostname
echo ""

export PYTHONPATH="/scratch2/mdeniaud/phd/pybdsim:$PYTHONPATH"
export PYTHONPATH="/scratch2/mdeniaud/phd/pymad8:$PYTHONPATH"
export PYTHONPATH="$PYTHONPATH:/scratch2/mdeniaud/phd/luxe-backgrounds/06_analysis"

# run bdsim and python analysis script to create the histogram root file
/scratch2/mdeniaud/phd/luxe-backgrounds/03_bdsimModel/bdsimAnalysisJob_wire.py -i ${INPUTFILE} -p ${NGEN} -x ${OFFSET} -s ${SEEDVALUE} -o ${OUTPUTFILE} -m ${MODELDIR}

echo "job finished"
date