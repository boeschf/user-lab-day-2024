#!/bin/bash

# Synopsis
# run.sh <binary> <nodes> <tasks per node> [<time limit>]
# 
# tasks per node: should be 1 or 4

BIN=$(realpath $1)
NODES=$2
NTASKS_PER_NODE=$3
TIME_LIMIT=${4:-"01:00:00"}

# calculate resources from input
CPUS_PER_TASK=$(( 288 / NTASKS_PER_NODE ))
NTASKS=$(( NODES * NTASKS_PER_NODE ))

# set the cpu-bindings
# - 1 task per node: bind to all cores
# - 4 tasks per node: bind tasks to numa nodes
CPUBIND=$([ "$3" -eq 1 ] && echo "verbose" || echo "verbose,rank_ldom")

# launch wrapper script
LAUNCHER=$(realpath ./launch_wrapper)
# uenv to run with
UENV=$(realpath ./nccl/store.squashfs)
# jobreport tool
JOBREPORT=$(realpath ./jobreport)

# output
NODES_STR=$(printf "%04d" "$NODES")
NTASKS_STR=$(printf "%05d" "$NTASKS")
OUTDIR="logs"
POSTFIX="n-${NTASKS_STR}-N-${NODES_STR}"
PREFIX="${OUTDIR}/job-${POSTFIX}"

# schedule the job
sbatch <<EOT
#!/bin/bash

#SBATCH --job-name nccl-test
#SBATCH --output=${PREFIX}-%j.out
#SBATCH --time=${TIME_LIMIT}
#SBATCH --nodes=${NODES}
#SBATCH --ntasks-per-node=${NTASKS_PER_NODE}
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --exclusive
#SBATCH --no-requeue

set -x

export OMP_NUM_THREADS=${CPUS_PER_TASK}

mkdir -p ${OUTDIR}
REPORT_DIR="${PREFIX}-\${SLURM_JOB_ID}.report"

# uncomment for pytorch/nccl debug logging
#export ENABLE_LOGGING=1

http_proxy=http://proxy.cscs.ch:8080 https_proxy=https://proxy.cscs.ch:8080 \
srun -u -l \
    --cpu-bind=${CPUBIND} \
    --uenv="${UENV}" \
    ${JOBREPORT} -o \${REPORT_DIR} -- \
    ${LAUNCHER} ${BIN}

${JOBREPORT} print \${REPORT_DIR}

EOT
