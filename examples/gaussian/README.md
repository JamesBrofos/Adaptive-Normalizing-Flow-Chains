## Violations of Stationarity with Non-Independent Adaptations

To visualize violations of stationarity, execute:
```
dSQ -C cascadelake --jobfile joblist.txt -p day --max-jobs 10000 -c 1 -t 24:00:00 --job-name stationarity -o output/stationarity-%A-%J.log --suppress-stats-file --submit
```
