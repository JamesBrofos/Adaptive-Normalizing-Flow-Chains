job = "source $HOME/.bashrc ; source activate adaptive-devel ; python experiment.py --{} --num-steps {} --num-trials {} --seed {}"
total_trials = 1000000
num_jobs_by_num_steps = {
    10: 10,
    100: 100,
    1000: 100,
    10000: 1000
}
with open('joblist.txt', 'w') as f:
    for v in ['no-violate', 'violate']:
        for num_steps in [10, 100, 1000, 10000]:
            num_jobs = num_jobs_by_num_steps[num_steps]
            num_trials = int(total_trials / num_jobs)
            for seed in range(num_jobs):
                f.write(job.format(v, num_steps, num_trials, seed) + '\n')
