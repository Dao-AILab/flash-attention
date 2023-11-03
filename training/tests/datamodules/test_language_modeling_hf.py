import os
from pathlib import Path
current_dir = Path(__file__).parent.absolute()


import pytest

import torch

import dotenv

from src.datamodules.language_modeling_hf import LMDataModule

# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv(override=True)


def div_up(x: int, y: int) -> int:
    return (x + y - 1) // y


# https://stackoverflow.com/questions/1006289/how-to-find-out-the-number-of-cpus-using-python/55423170#55423170
def num_cpu_cores():
    try:
        import psutil
        return psutil.cpu_count(logical=False)
    except ImportError:
        return len(os.sched_getaffinity(0))


class TestLMDataModule:

    def test_wikitext2(self):
        batch_size = 7
        dataset_name = 'wikitext'
        dataset_config_name = 'wikitext-2-raw-v1'
        data_dir = Path(os.getenv('DATA_DIR', current_dir.parent.parent / 'data'))
        cache_dir = data_dir / 'wikitext-2' / 'cache'
        max_length = 1024
        datamodule = LMDataModule(dataset_name, tokenizer_name='gpt2',
                                  dataset_config_name=dataset_config_name,
                                  max_length=max_length, cache_dir=cache_dir,
                                  add_eos=False, batch_size=batch_size, num_workers=4)
        datamodule.prepare_data()
        datamodule.setup(stage='fit')
        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()
        datamodule.setup(stage='test')
        test_loader = datamodule.test_dataloader()
        train_len = 2391884
        val_len = 247289
        test_len = 283287
        assert len(train_loader) == div_up((train_len - 1) // max_length, batch_size)
        assert len(val_loader) == div_up((val_len - 1) // max_length, batch_size)
        assert len(test_loader) == div_up((test_len - 1) // max_length, batch_size)
        for loader in [train_loader, val_loader, test_loader]:
            x, y = next(iter(loader))
            assert x.dim() == 2
            assert x.shape == (batch_size, max_length)
            assert x.dtype == torch.long
            assert torch.allclose(x[:, 1:], y[:, :-1])

    def test_wikitext103(self):
        batch_size = 7
        dataset_name = 'wikitext'
        dataset_config_name = 'wikitext-103-raw-v1'
        data_dir = Path(os.getenv('DATA_DIR', current_dir.parent.parent / 'data'))
        cache_dir = data_dir / 'wikitext-103' / 'cache'
        max_length = 1024
        datamodule = LMDataModule(dataset_name, tokenizer_name='gpt2',
                                  dataset_config_name=dataset_config_name,
                                  max_length=max_length, cache_dir=cache_dir,
                                  add_eos=False, batch_size=batch_size, num_workers=4)
        datamodule.prepare_data()
        datamodule.setup(stage='fit')
        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()
        datamodule.setup(stage='test')
        test_loader = datamodule.test_dataloader()
        train_len = 117920140
        val_len = 247289
        test_len = 283287
        assert len(train_loader) == div_up((train_len - 1) // max_length, batch_size)
        assert len(val_loader) == div_up((val_len - 1) // max_length, batch_size)
        assert len(test_loader) == div_up((test_len - 1) // max_length, batch_size)
        for loader in [train_loader, val_loader, test_loader]:
            x, y = next(iter(loader))
            assert x.dim() == 2
            assert x.shape == (batch_size, max_length)
            assert x.dtype == torch.long
            assert torch.allclose(x[:, 1:], y[:, :-1])

    def test_openwebtext(self):
        batch_size = 8
        dataset_name = 'openwebtext'
        dataset_config_name = None
        data_dir = Path(os.getenv('DATA_DIR', current_dir.parent.parent / 'data'))
        cache_dir = data_dir / 'openwebtext' / 'cache'
        max_length = 1024
        datamodule = LMDataModule(dataset_name, tokenizer_name='gpt2',
                                  dataset_config_name=dataset_config_name,
                                  max_length=max_length, cache_dir=cache_dir,
                                  add_eos=True, batch_size=batch_size,
                                  num_workers=num_cpu_cores() // 2)
        datamodule.prepare_data()
        datamodule.setup(stage='fit')
        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()
        datamodule.setup(stage='test')
        test_loader = datamodule.test_dataloader()
        train_len = 9035582198
        val_len = 4434897
        test_len = 4434897
        assert len(train_loader) == div_up((train_len - 1) // max_length, batch_size)
        assert len(val_loader) == div_up((val_len - 1) // max_length, batch_size)
        assert len(test_loader) == div_up((test_len - 1) // max_length, batch_size)
        for loader in [train_loader, val_loader, test_loader]:
            x, y = next(iter(loader))
            assert x.dim() == 2
            assert x.shape == (batch_size, max_length)
            assert x.dtype == torch.long
            assert torch.allclose(x[:, 1:], y[:, :-1])

    def test_lambada(self):
        batch_size = 8
        dataset_name = 'lambada'
        dataset_config_name = None
        data_dir = Path(os.getenv('DATA_DIR', current_dir.parent.parent / 'data'))
        cache_dir = data_dir / 'lambada' / 'cache'
        max_length = 1024
        datamodule = LMDataModule(dataset_name, tokenizer_name='gpt2',
                                  dataset_config_name=dataset_config_name,
                                  max_length=max_length, cache_dir=cache_dir,
                                  add_eos=True, batch_size=batch_size,
                                  num_workers=64)
        datamodule.prepare_data()
        datamodule.setup(stage='fit')
        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()
        datamodule.setup(stage='test')
        test_loader = datamodule.test_dataloader()
        train_len = 9035582198
        val_len = 4434897
        test_len = 4434897
        assert len(train_loader) == div_up((train_len - 1) // max_length, batch_size)
        assert len(val_loader) == div_up((val_len - 1) // max_length, batch_size)
        assert len(test_loader) == div_up((test_len - 1) // max_length, batch_size)
        for loader in [train_loader, val_loader, test_loader]:
            x, y = next(iter(loader))
            assert x.dim() == 2
            assert x.shape == (batch_size, max_length)
            assert x.dtype == torch.long
            assert torch.allclose(x[:, 1:], y[:, :-1])

    def test_the_pile(self):
        batch_size = 8
        dataset_name = 'the_pile'
        dataset_config_name = None
        data_dir = Path(os.getenv('DATA_DIR', current_dir.parent.parent / 'data'))
        cache_dir = data_dir / 'the_pile' / 'cache'
        max_length = 2048
        # Dataset is too large to fit into memory, need to use disk for concatenation
        datamodule = LMDataModule(dataset_name, tokenizer_name='gpt2',
                                  dataset_config_name=dataset_config_name,
                                  max_length=max_length, cache_dir=cache_dir,
                                  add_eos=True, batch_size=batch_size,
                                  num_workers=num_cpu_cores() // 2, use_shmem=False)
        datamodule.prepare_data()
        datamodule.setup(stage='fit')
        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()
        datamodule.setup(stage='test')
        test_loader = datamodule.test_dataloader()
        train_len = 374337375694
        val_len = 383326395
        test_len = 373297018
        assert len(train_loader) == div_up((train_len - 1) // max_length, batch_size)
        assert len(val_loader) == div_up((val_len - 1) // max_length, batch_size)
        assert len(test_loader) == div_up((test_len - 1) // max_length, batch_size)
        for loader in [train_loader, val_loader, test_loader]:
            x, y = next(iter(loader))
            assert x.dim() == 2
            assert x.shape == (batch_size, max_length)
            assert x.dtype == torch.long
            assert torch.allclose(x[:, 1:], y[:, :-1])

    def test_pg19(self):
        batch_size = 8
        dataset_name = 'pg19'
        dataset_config_name = None
        data_dir = Path(os.getenv('DATA_DIR', current_dir.parent.parent / 'data'))
        cache_dir = data_dir / 'pg19' / 'cache'
        max_length = 2048
        # Dataset is too large to fit into memory, need to use disk for concatenation
        datamodule = LMDataModule(dataset_name, tokenizer_name='gpt2',
                                  dataset_config_name=dataset_config_name,
                                  max_length=max_length, cache_dir=cache_dir,
                                  add_eos=True, batch_size=batch_size,
                                  num_workers=num_cpu_cores() // 2)
        datamodule.prepare_data()
        datamodule.setup(stage='fit')
        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()
        datamodule.setup(stage='test')
        test_loader = datamodule.test_dataloader()
        train_len = 3066544128
        val_len = 4653056
        test_len = 10584064
        assert len(train_loader) == div_up((train_len - 1) // max_length, batch_size)
        assert len(val_loader) == div_up((val_len - 1) // max_length, batch_size)
        assert len(test_loader) == div_up((test_len - 1) // max_length, batch_size)
        for loader in [train_loader, val_loader, test_loader]:
            x, y = next(iter(loader))
            assert x.dim() == 2
            assert x.shape == (batch_size, max_length)
            assert x.dtype == torch.long
            assert torch.allclose(x[:, 1:], y[:, :-1])
