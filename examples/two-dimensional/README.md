# Sampling with Normalizing Flows in Two-Dimensional Distributions

Samples from a Gaussian mixture with nine modes, Neal's funnel distribution, a Student-t distribution with fifteen degrees-of-freedom, and banana-shaped distribution with long, thin tails. One can execute the experiments as follows:
```
dSQ -C cascadelake --jobfile joblist.txt -p day --max-jobs 1000 -c 2 -t 24:00:00 --job-name adaptive -o output/adaptive-%A-%J.log --suppress-stats-file --submit
```
