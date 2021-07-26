#!/bin/bash
#SBATCH --partition day
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --mem-per-cpu 8G
#SBATCH --time 1-0:00:00
#SBATCH --job-name tensorboard
#SBATCH --output tensorboard-%J.log

# get tunneling info
XDG_RUNTIME_DIR=""
# if we need a random port we can use this.
# port=$(shuf -i8000-9999 -n1)
# but I think a fixed port will suffice.
port=8008

node=$(hostname -s)
user=$(whoami)
cluster=$(hostname -f | awk -F"." '{print $2}')

# print tunneling instructions
echo -e "
MacOS or linux terminal command to create your ssh tunnel
ssh -N -L ${port}:${node}:${port} ${user}@${cluster}.hpc.yale.edu

Windows MobaXterm info
Forwarded port:same as remote port
Remote server: ${node}
Remote port: ${port}
SSH server: ${cluster}.hpc.yale.edu
SSH login: $user
SSH port: 22

Use a Browser on your local machine to go to:
localhost:${port}  (prefix w/ https:// if using password)
"

# load modules or conda environments here

source $HOME/.bashrc
source activate adaptive-devel

tensorboard --logdir logs --port=${port} --host=${node}
