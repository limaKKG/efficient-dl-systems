import os
import typing as tp
import boto3
from botocore.client import Config

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from utils import Clothes, get_labels_dict


class ClothesDataset(Dataset):
    def __init__(self, folder_path, frame, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.frame = frame.set_index("image")
        self.img_list = list(self.frame.index.values)

        self.label2ix = get_labels_dict()

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        img = Image.open(f"{self.folder_path}/{img_name}.jpg").convert("RGB")
        img_transformed = self.transform(img)
        label = self.label2ix[self.frame.loc[img_name]["label"]]

        return img_transformed, label


def download_extract_dataset():
    target_dir = f"{Clothes.directory}/{Clothes.train_val_img_dir}"
    if os.path.exists(target_dir):
        print("Dataset already extracted")
        return

    os.makedirs(Clothes.directory, exist_ok=True)
    os.makedirs(target_dir, exist_ok=True)

    bucket = "rnd-customer-service-platform-ml-data"
    csv_key = "clothing_dataset/images.csv"
    original_prefix = "clothing_dataset/images_original/"
    compressed_prefix = "clothing_dataset/images_compressed/"

    s3 = boto3.client(
        "s3",
        aws_access_key_id=os.environ["access_key_id"],
        aws_secret_access_key=os.environ["secret_access_key"],
        endpoint_url="https://s3-msk.tinkoff.ru",
        config=Config(signature_version="s3v4"),
    )

    # CSV
    csv_path = f"{Clothes.directory}/{Clothes.csv_name}"
    if not os.path.exists(csv_path):
        s3.download_file(bucket, csv_key, csv_path)

    def _download_prefix(prefix: str, dst: str):
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith("/"):
                    continue
                filename = os.path.basename(key)
                local_path = os.path.join(dst, filename)
                if not os.path.exists(local_path):
                    os.makedirs(os.path.dirname(local_path), exist_ok=True)
                    s3.download_file(bucket, key, local_path)

    _download_prefix(original_prefix, target_dir)
    # при желании сохраним сжатые версии отдельно
    compressed_dir = f"{Clothes.directory}/images_compressed"
    os.makedirs(compressed_dir, exist_ok=True)
    _download_prefix(compressed_prefix, compressed_dir)


def get_train_transforms() -> tp.Any:
    return transforms.Compose(
        [
            transforms.Resize((320, 320)),
            transforms.CenterCrop(224),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.AugMix(),
            transforms.ToTensor(),
        ]
    )


def get_val_transforms() -> tp.Any:
    return transforms.Compose(
        [
            transforms.Resize((320, 320)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )
