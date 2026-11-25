"""统一的数据路径、特征读取和批次加载接口。"""
from pathlib import Path
import pickle

import torch
from torch.utils.data import DataLoader, Dataset

# 只需修改这一处默认数据父目录；命令行 --data_root 优先于此设置。
DATA_ROOT = Path(r"D:\论文\ICME\CLIP\CLIP")
DATASETS = ("mirflickr", "nus-wide", "mscoco")
SPLITS = ("train", "query", "retrieval")


class CustomDataSet(Dataset):
    def __init__(self, images, texts, labs):
        self.images, self.texts, self.labs = images, texts, labs

    def __getitem__(self, index):
        return self.images[index], self.texts[index], self.labs[index], index

    def __len__(self):
        return len(self.texts)


def resolve_data_root(data_root=None):
    return Path(DATA_ROOT if data_root is None else data_root).expanduser().resolve()


def split_paths(dataset, data_root=None):
    if dataset not in DATASETS:
        raise ValueError("未知数据集：" + str(dataset))
    root = resolve_data_root(data_root) / dataset
    return {name: root / (name + ".pkl") for name in SPLITS}


def read_split(file, limit=None):
    file = Path(file)
    if not file.is_file():
        raise FileNotFoundError(
            f"找不到数据文件：{file}。请修改 load_dataset.py 中的 DATA_ROOT，"
            "或使用 --data_root 指定包含 mirflickr、nus-wide、mscoco 的父目录。")
    if limit is not None and (not isinstance(limit, int) or limit < 1):
        raise ValueError("样本上限必须为正整数或 None")
    with file.open("rb") as stream:
        data = pickle.load(stream)
    tensors = tuple(torch.tensor(data[key][:limit] if limit else data[key], dtype=dtype)
                    for key, dtype in (("image", torch.float32), ("text", torch.float32), ("label", torch.int64)))
    if len({len(x) for x in tensors}) != 1 or len(tensors[0]) == 0:
        raise ValueError(f"数据为空或图文标签未对齐：{file}")
    if (tensors[0].ndim != 2 or tensors[1].ndim != 2 or tensors[2].ndim != 2 or
            tensors[0].shape[1] != 512 or tensors[1].shape[1] != 512):
        raise ValueError(f"需要 512 维图像/文本特征及二维多标签数组：{file}")
    return tensors


def read_dataset(dataset, data_root=None, train_limit=4992, query_limit=None, retrieval_limit=None):
    limits = dict(train=train_limit, query=query_limit, retrieval=retrieval_limit)
    return {name: read_split(file, limits[name]) for name, file in split_paths(dataset, data_root).items()}


def make_loader(tensors, batch_size, shuffle=False, seed=0, protocol="standard"):
    if protocol not in ("standard", "legacy"):
        raise ValueError("未知批次协议：" + str(protocol))
    generator = torch.Generator().manual_seed(seed) if protocol == "standard" else None
    return DataLoader(CustomDataSet(*tensors), batch_size=batch_size, shuffle=shuffle,
                      drop_last=False, num_workers=0, pin_memory=torch.cuda.is_available(), generator=generator)


def make_loaders(data, batch_size, seed=2023, protocol="legacy"):
    names = SPLITS if protocol == "legacy" else tuple(data)
    return {name: make_loader(data[name], batch_size, shuffle=name == "train", seed=seed + index,
                              protocol=protocol) for index, name in enumerate(names)}


def load_dataset(dataset, batch_size, data_root=None, train_limit=4992, query_limit=None,
                 retrieval_limit=None, seed=2023, protocol="legacy"):
    data = read_dataset(dataset, data_root, train_limit, query_limit, retrieval_limit)
    return make_loaders(data, batch_size, seed, protocol)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, default="mirflickr")
    parser.add_argument("--data_root", type=Path)
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()
    loaders = load_dataset(args.dataset, args.batch_size, args.data_root)
    print("数据父目录：", resolve_data_root(args.data_root))
    for name, loader in loaders.items():
        print(name, "样本数：", len(loader.dataset), "批次数：", len(loader))
