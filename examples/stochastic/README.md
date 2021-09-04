# Sampling with Normalizing Flows in a Stochastic Differential Equation

Samples from an Ornstein-Uhlenbeck stochastic differential equation model.
```
dSQ -C cascadelake --jobfile joblist.txt -p day --mem-per-cpu 20GB --max-jobs 1000 -c 5 -t 24:00:00 --job-name adaptive -o output/adaptive-%A-%J.log --suppress-stats-file --submit
```
