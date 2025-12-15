from run_epoch import run_epoch, DataMode
import pandas as pd
import boto3 
import os
import torch 
from pathlib import Path

def main():
    target_dir = Path('wikitext-103-raw-v1')
    target_dir.mkdir(parents=True, exist_ok=True)

    s3 = boto3.client(
        's3',
        aws_access_key_id=os.environ["access_key_id"],
        aws_secret_access_key=os.environ["secret_access_key"],
        endpoint_url='https://s3-msk.tinkoff.ru'
    )  
    s3.download_file('rnd-customer-service-platform-ml-data', 'wikitext-103-raw-v1/train-00000-of-00002.txt', 'wikitext-103-raw-v1/train-00000-of-00002.txt')
    s3.download_file('rnd-customer-service-platform-ml-data', 'wikitext-103-raw-v1/train-00001-of-00002.txt', 'wikitext-103-raw-v1/train-00001-of-00002.txt')
    BATCH_SIZE = 8
    K_LIST = [1, 5, 10, 20, 50]
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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

if __name__ == "__main__":
    main()