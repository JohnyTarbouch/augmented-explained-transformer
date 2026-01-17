from datasets import load_dataset


def load_sst2(cache_dir: str):
    return load_dataset("glue", "sst2", cache_dir=cache_dir)
