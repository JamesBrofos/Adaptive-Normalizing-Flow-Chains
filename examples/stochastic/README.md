# Sampling with Adaptive MCMC in a Brownian Bridge

Samples from a non-central Brownian bridge. One can execute the experiments as follows:
```
dSQ -C cascadelake --jobfile joblist.txt -p day --max-jobs 1000 -c 2 -t 24:00:00 --job-name adaptive -o output/adaptive-%A-%J.log --suppress-stats-file --submit
```
