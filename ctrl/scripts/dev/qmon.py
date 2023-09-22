"""Script to monitor slurm using squeue and not submit more jobs than any given partition supports"""
import os
from collections import defaultdict
import subprocess as sp
from time import sleep

from remake.util import sysrun
from remake import load_remake
from remake.executor.slurm_executor import SlurmExecutor

# Things don't work if I don't do this!
os.chdir('../../remakefiles')

e5p = load_remake('era5_process.py')

# https://help.jasmin.ac.uk/article/4881-lotus-queues
# Queue name      Max run time    Default run time    Max CPU cores
# per job     Max CpuPer
# UserLimit       Priority
# test                                             4 hrs      1hr     8   8   30
# short-serial    24 hrs      1hr     1       2000    30
# par-single  48 hrs      1hr     16      300     25
# par-multi   48 hrs      1hr     256     300     20
# long-serial     168 hrs     1hr     1   300     10
# high-mem    48 hrs      1hr     1   75      30
# short-serial-4hr (Note 3)   4 hrs   1hr     1   1000        30
partition_max_jobs = {
    'short-serial': 2000,
    'par-single': 300,
    'par-multi': 300,
    'long-serial': 300,
    'high-mem': 75,
    'short-serial-4hr': 1000,
}

def get_squeue_output():
    try:
        # get jobid, partition and job name.
        # job name is 10 character task key.
        output = sysrun('squeue -u mmuetz -o "%.18i %.20P %.10j %.3t"').stdout
        print(output.strip())
    except sp.CalledProcessError as cpe:
        print('Error on squeue command')
        print(cpe)
        print('===ERROR===')
        print(cpe.stderr)
        print('===ERROR===')
        raise

    return output

# Make sure I can handle dependencies.
e5p.task_ctrl.build_task_DAG()

# Use a SlurmExecutor to submit the jobs.
slurm_config = e5p.config.get('slurm', {})
ex = SlurmExecutor(e5p.task_ctrl, slurm_config)

remaining_tasks = [t for t in e5p.tasks]
while remaining_tasks:
    # Scan current slurm jobs, getting info about how many in each partition.
    output = get_squeue_output()
    partition_jobs = defaultdict(list)
    for line in output.split('\n')[1:]:
        if not line:
            continue
        jobid, squeue_partition, task_key, status = line.split()
        partition_jobs[squeue_partition].append(task_key)

    enqueued_tasks = []
    for task in remaining_tasks:
        # Figure out the task's partition.
        task_config = getattr(task, 'config', {})
        if 'slurm' in task_config and 'partition' in task_config['slurm']:
            task_partition = task_config['slurm']['partition']
        elif slurm_config and 'partition' in slurm_config:
            task_partition = slurm_config['partition']
        else:
            task_partition = 'short-serial'

        # Enqueue tasks (i.e. submit them to SLURM) if there is space on the partition.
        if len(partition_jobs[task_partition]) < partition_max_jobs[task_partition]:
            ex.enqueue_task(task)
            enqueued_tasks.append(task)
            partition_jobs[task_partition].append(task.path_hash_key())

    for task in enqueued_tasks:
        remaining_tasks.remove(task)
    print('remaining tasks:', len(remaining_tasks))
    sleep(60)
