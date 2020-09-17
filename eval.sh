#!/bin/bash
#
#SBATCH -p All # partition (queue)
#SBATCH -N 1 # number of nodes
#SBATCH -J ASIL # Job Name
#SBATCH -o slurm/%N.%j.out # STDOUT
#SBATCH -e slurm/%N.%j.err # STDERR
#SBATCH --mail-type=ALL
#SBATCH --mail-user=philippaltmann@deap.io
for run in 11 #1 #2 3 4
do
  for samples in 16 #4 8 16 32 64 128 #256 512 1024
  do
    for update in 1 #2 4 8 0
    do
      # Sil Mixture
      for alpha in 0.0 0.2 0.4 0.6 0.8 1.0 #1
      do
	# echo 'srun -p All -J="$1 $alpha $update $samples Run $run" -N 1 xvfb-run -s "-screen 0 1400x900x24" python3 evaluate.py $1 $alpha $update $samples $run T &'
        # echo "xvfb-run -s '-screen 0 1400x900x24' python3 evaluate.py $1 $alpha $update $samples T"
        srun -J "$1 $alpha $update $samples Run $run" -p All  xvfb-run -s '-screen 0 1400x900x24' python3 evaluate.py $1 $alpha $update $samples $run T &
        # srun -J "$1 $alpha $update $samples Run $run" -p All  xvfb-run -s '-screen 0 1400x900x24' python3 evaluate.py $1 $alpha $update $samples $run T &
      done
      # echo ""
      wait
    done
    # wait
  done
done
wait
