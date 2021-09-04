# Sampling with Normalizing Flows in Regression Posteriors

Samples from a Bayesian linear and logistic regression models. One can execute the experiments as follows:
```
dSQ -C cascadelake --jobfile joblist.txt -p day --max-jobs 1000 -c 5 -t 24:00:00 --job-name adaptive -o output/adaptive-%A-%J.log --suppress-stats-file --submit
```
