source suite.sh
for var in a b c d e f g j k m; do moo ls moose:/crum/${SUITE}/ap${var}.pp|tee -a ${SUITE}_ls_files.txt; done
