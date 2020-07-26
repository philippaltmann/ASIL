#!/bin/bash
# Samples in Buffer
for samples in 512  ### Outer for loop ###
do
  # Sil Mixture
  for alpha in 0.0 1.0  ### Outer for loop ###
  do
    echo "sbatch --partition=All -J='ASIL:$1-$alpha-$samples' xvfb-run -s '-screen 0 1400x900x24' python3 evaluate.py $1 $alpha $samples T"
    sbatch --partition=All -J='ASIL:$1-$alpha-$samples' xvfb-run -s '-screen 0 1400x900x24' python3 evaluate.py $1 $alpha $samples T
    #xvfb-run -s "-screen 0 1400x900x24" python3
    # train_dqn.sh 0 $step_penalty; done
    done
  echo "" #### print the new line ###
done
