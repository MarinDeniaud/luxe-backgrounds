condor_submit htcondor_wire.sub filename=T20_for_wire_with_offset_-0.50_bias_5e0.gmad seedstart=0 npart=1000 njobs=100
condor_submit htcondor_wire.sub filename=T20_for_wire_with_offset_-0.45_bias_5e0.gmad seedstart=0 npart=1000 njobs=100
condor_submit htcondor_wire.sub filename=T20_for_wire_with_offset_-0.40_bias_5e0.gmad seedstart=0 npart=1000 njobs=100
condor_submit htcondor_wire.sub filename=T20_for_wire_with_offset_-0.35_bias_5e0.gmad seedstart=0 npart=1000 njobs=100
condor_submit htcondor_wire.sub filename=T20_for_wire_with_offset_-0.30_bias_5e0.gmad seedstart=0 npart=1000 njobs=100
condor_submit htcondor_wire.sub filename=T20_for_wire_with_offset_-0.25_bias_5e0.gmad seedstart=0 npart=1000 njobs=100
condor_submit htcondor_wire.sub filename=T20_for_wire_with_offset_-0.20_bias_5e0.gmad seedstart=0 npart=1000 njobs=100
condor_submit htcondor_wire.sub filename=T20_for_wire_with_offset_-0.15_bias_5e0.gmad seedstart=0 npart=1000 njobs=100
condor_submit htcondor_wire.sub filename=T20_for_wire_with_offset_-0.10_bias_5e0.gmad seedstart=0 npart=1000 njobs=100
condor_submit htcondor_wire.sub filename=T20_for_wire_with_offset_-0.05_bias_5e0.gmad seedstart=0 npart=1000 njobs=100
condor_submit htcondor_wire.sub filename=T20_for_wire_with_offset_+0.00_bias_5e0.gmad seedstart=0 npart=1000 njobs=100
condor_submit htcondor_wire.sub filename=T20_for_wire_with_offset_+0.05_bias_5e0.gmad seedstart=0 npart=1000 njobs=100
condor_submit htcondor_wire.sub filename=T20_for_wire_with_offset_+0.10_bias_5e0.gmad seedstart=0 npart=1000 njobs=100
condor_submit htcondor_wire.sub filename=T20_for_wire_with_offset_+0.15_bias_5e0.gmad seedstart=0 npart=1000 njobs=100
condor_submit htcondor_wire.sub filename=T20_for_wire_with_offset_+0.20_bias_5e0.gmad seedstart=0 npart=1000 njobs=100
condor_submit htcondor_wire.sub filename=T20_for_wire_with_offset_+0.25_bias_5e0.gmad seedstart=0 npart=1000 njobs=100
condor_submit htcondor_wire.sub filename=T20_for_wire_with_offset_+0.30_bias_5e0.gmad seedstart=0 npart=1000 njobs=100
condor_submit htcondor_wire.sub filename=T20_for_wire_with_offset_+0.35_bias_5e0.gmad seedstart=0 npart=1000 njobs=100
condor_submit htcondor_wire.sub filename=T20_for_wire_with_offset_+0.40_bias_5e0.gmad seedstart=0 npart=1000 njobs=100
condor_submit htcondor_wire.sub filename=T20_for_wire_with_offset_+0.45_bias_5e0.gmad seedstart=0 npart=1000 njobs=100
condor_submit htcondor_wire.sub filename=T20_for_wire_with_offset_+0.50_bias_5e0.gmad seedstart=0 npart=1000 njobs=100

#while read tag; do condor_submit ../03_bdsimModel/htcondor_wire.sub filename=../03_bdsimModel/"$tag".gmad seedstart=0 npart=1000 njobs=100; done < ../06_analysis/tagfilelistwire