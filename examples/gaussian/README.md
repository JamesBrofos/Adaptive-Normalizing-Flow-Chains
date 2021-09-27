## Violations of Stationarity with Non-Independent Adaptations

To visualize violations of stationarity, execute:
```
dSQ -C cascadelake --jobfile joblist.txt -p day --max-jobs 1000 -c 5 -t 24:00:00 --job-name stationarity -o output/stationarity-%A-%J.log --suppress-stats-file --submit
```
