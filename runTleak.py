wq sub -r "N:204; threads:1;hostfile:auto" -c "a_mpirun -hostfile %hostfile% ./Prepare_Tleak.py Q 100"
wq sub -r "N:204; threads:1;hostfile:auto" -c "a_mpirun -hostfile %hostfile% ./Prepare_Tleak.py U 100"
wq sub -r "N:204; threads:1;hostfile:auto" -c "a_mpirun -hostfile %hostfile% ./Prepare_Tleak.py Q 143"
wq sub -r "N:204; threads:1;hostfile:auto" -c "a_mpirun -hostfile %hostfile% ./Prepare_Tleak.py U 143"
