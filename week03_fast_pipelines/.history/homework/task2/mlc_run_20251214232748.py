from run_epoch import run_epoch, DataMode
import pandas as pd
import boto3 

BATCH_SIZE = 8
K_LIST = [1, 5, 10, 20, 50]
DEVICE = "cuda" 

results = []
for mode in [DataMode.BRAIN, DataMode.BIG_BRAIN]:
    stats = run_epoch(mode, batch_size=BATCH_SIZE, device=DEVICE)
    stats["mode"] = mode.name
    stats["k"] = None
    results.append(stats)

for k in K_LIST:
    stats = run_epoch(DataMode.ULTRA_BIG_BRAIN, batch_size=BATCH_SIZE, k=k, device=DEVICE)
    stats["mode"] = DataMode.ULTRA_BIG_BRAIN.name
    stats["k"] = k
    results.append(stats)

stats = run_epoch(DataMode.ULTRA_DUPER_BIG_BRAIN, batch_size=1, device=DEVICE)
stats["mode"] = DataMode.ULTRA_DUPER_BIG_BRAIN.name
stats["k"] = None
results.append(stats)

df = pd.DataFrame(results)

df.to_csv('evaluation_results.csv', index=False)