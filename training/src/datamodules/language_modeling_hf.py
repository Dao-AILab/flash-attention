# Adapted from https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_clm.py
from itertools import chain
from pathlib import Path
import pickle
from typing import Any, List, Union
import subprocess
import mmap

from multiprocessing.shared_memory import SharedMemory

import numpy as np

import torch
from torch.utils.data.dataloader import DataLoader, Dataset
from transformers import AutoTokenizer
from datasets import load_dataset

from pytorch_lightning import LightningDataModule

from src.datamodules.datasets.lm_dataset import LMDataset
from src.datamodules.fault_tolerant_sampler import RandomFaultTolerantSampler
from src.datamodules.fault_tolerant_sampler import FaultTolerantDistributedSampler
from src.datamodules.datasets.detokenizer import DATASET_TOKENIZATION_REGISTRY
from src.utils.utils import get_logger
logger = get_logger()


# https://github.com/numpy/numpy/issues/18294
class SHMArray(np.ndarray): #copied from https://numpy.org/doc/stable/user/basics.subclassing.html#slightly-more-realistic-example-attribute-added-to-existing-array

    def __new__(cls, input_array, shm=None):
        obj = np.asarray(input_array).view(cls)
        obj.shm = shm
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.shm = getattr(obj, 'shm', None)


class LMDataModule(LightningDataModule):
    def __init__(self, dataset_name, tokenizer_name, dataset_config_name=None, max_length=1024,
                 cache_dir=None, val_ratio=0.0005, val_split_seed=2357, add_eos=True,
                 detokenize=False, val_only=False, batch_size=32, batch_size_eval=None, num_workers=1,
                 shuffle=False, pin_memory=False, drop_last=False, fault_tolerant=False, ddp=False,
                 fast_forward_epochs=None, fast_forward_batches=None,
                 use_shmem=True):
        super().__init__()
        self.dataset_name = dataset_name
        self.dataset_config_name = dataset_config_name
        self.tokenizer_name = tokenizer_name
        self.cache_dir = None if cache_dir is None else Path(cache_dir).expanduser()
        self.max_length = max_length
        self.val_ratio = val_ratio
        self.val_split_seed = val_split_seed
        self.val_only = val_only
        self.add_eos = add_eos
        self.detokenize = detokenize
        self.batch_size = batch_size
        self.batch_size_eval = batch_size_eval if batch_size_eval is not None else self.batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        if fault_tolerant:
            assert self.shuffle
        self.fault_tolerant = fault_tolerant
        if ddp:
            assert fault_tolerant
        self.ddp = ddp
        self.fast_forward_epochs = fast_forward_epochs
        self.fast_forward_batches = fast_forward_batches
        if self.fast_forward_epochs is not None or self.fast_forward_batches is not None:
            assert ddp and fault_tolerant

        self.use_shmem = use_shmem
        if self.use_shmem:
            assert cache_dir is not None

    def prepare_data(self):
        if self.cache_dir is None:  # Just download the dataset
            load_dataset(self.dataset_name, self.dataset_config_name)
        else:  # Process the dataset and save it
            self.process_dataset()

    def setup(self, stage=None):
        if stage == 'test' and hasattr(self, 'dataset_test'):
            return
        concat_ids, self.tokenizer = self.process_dataset()
        self.vocab_size = len(self.tokenizer)
        # Create all splits
        self.dataset_train, self.dataset_val, self.dataset_test = [
            LMDataset(concat_ids[split], seq_len=self.max_length)
            for split in ['train', 'validation', 'test']
        ]

    def process_dataset(self):
        cache_dir = None if self.cache_dir is None else self.cache_dir / self._cache_dir_name
        if cache_dir is not None:
            if cache_dir.is_dir():
                return self._load_from_cache(cache_dir)

        raw_datasets = load_dataset(self.dataset_name, self.dataset_config_name)
        # https://github.com/stanford-crfm/mistral/blob/main/src/corpora/auto.py
        if 'validation' not in raw_datasets:
            assert "train" in raw_datasets, "You must have train in raw_datasets to make a validation raw_datasets"
            raw_datasets = raw_datasets["train"].train_test_split(
                test_size=self.val_ratio, seed=self.val_split_seed,
                shuffle=True  # Otherwise test will be at the end of the dataset
            )
            raw_datasets['validation'] = raw_datasets['test']

        if self.val_only:  # Should only be used for evaluation, not for training
            raw_datasets['train'] = raw_datasets['validation']

        # [2021-12-25] TD: Running the detokenizer on wikitext-103 makes ppl worse
        # (GPT2-small val ppl after 10 epochs ~22 -> ~25)
        # However, it's useful for zero-shot transfer from Openwebtext,
        # as after detokenization it's closer to Openwebtext's format.
        # https://github.com/stanford-crfm/mistral/issues/12
        if self.detokenize:
            if self.dataset_name in DATASET_TOKENIZATION_REGISTRY:
                detokenizer = DATASET_TOKENIZATION_REGISTRY[self.dataset_name]
                raw_datasets = raw_datasets.map(
                    lambda example: {'text': detokenizer(example['text'])},
                    num_proc=max(self.num_workers, 1),
                    desc='Running detokenizer on dataset'
                )

        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, use_fast=True)
        # Preprocessing the datasets.
        # First we tokenize all the texts.
        column_names = raw_datasets["train"].column_names
        text_column_name = "text" if "text" in column_names else column_names[0]
        # [2021-12-25] TD: For wikitext, don't need to add the EOS since each example already ends
        # with '\n', and there are no other '\n' in the examples.
        # assert all([t.count('\n') == 1 for t in raw_datasets['train']['text'] if t])
        # Add EOS token to the end of the text if the text is not empty
        # https://github.com/stanford-crfm/mistral/issues/91
        # https://github.com/stanford-crfm/mistral/pull/98
        if self.add_eos:
            add_eos = lambda seq: (seq + tokenizer.eos_token) if seq else seq
            add_eos_batched = lambda seqs: [add_eos(seq) for seq in seqs]
            tokenize = lambda example: tokenizer(add_eos_batched(example[text_column_name]))
        else:
            tokenize = lambda example: tokenizer(example[text_column_name])
        # tokenized_datasets = raw_datasets.map(
        #     tokenize,
        #     batched=True,
        #     num_proc=max(self.num_workers, 1),
        #     remove_columns=column_names,
        #     desc="Running tokenizer on dataset",
        # )
        dtype = np.uint16 if tokenizer.vocab_size < 64 * 1024 else np.int32
        def tokenize_concat(examples):
            # We just need 'input_ids', not 'attention_mask' (since it's all 1)
            input_ids = np.fromiter(chain(*tokenize(examples)['input_ids']), dtype=dtype)
            # Need to return a list since we're doing batched processing
            return {'input_ids': [input_ids], 'len': [len(input_ids)]}
        tokenized_datasets = raw_datasets.map(
            tokenize_concat,
            batched=True,
            num_proc=max(self.num_workers, 1),
            remove_columns=column_names,
            desc="Running tokenizer on dataset",
        )

        if self.use_shmem:
            # Concatenate all input_ids into an array in shared memory
            def write_ids_to_shm(example, shm_name, array_len):
                shm = SharedMemory(name=shm_name)
                shm_arr = np.ndarray((array_len,), dtype=dtype, buffer=shm.buf)
                start_idx = example['len_offset'] - len(example['input_ids'])
                shm_arr[start_idx:example['len_offset']] = example['input_ids']
                shm.close()
            concat_ids = {}
            for name, ds in tokenized_datasets.items():
                tokenized_datasets[name] = ds.add_column('len_offset', np.cumsum(ds['len']))
                array_len = tokenized_datasets[name][-1]['len_offset']
                shm = SharedMemory(create=True, size=array_len * np.dtype(dtype).itemsize)
                shm_name = shm.name
                tokenized_datasets[name].map(
                    write_ids_to_shm,
                    fn_kwargs={'shm_name': shm_name, 'array_len': array_len},
                    batched=False,
                    num_proc=max(self.num_workers, 1),
                    desc="Concatenating examples",
                )
                shm_arr = np.ndarray((array_len,), dtype=dtype, buffer=shm.buf)
                # We need to keep a reference to the shared memory, otherwise it gets garbage-collected
                # when it goes out of scope, and that memory is gone.
                # https://github.com/numpy/numpy/issues/18294
                concat_ids[name] = SHMArray(shm_arr, shm=shm)
        else:
            # Use disk
            concat_ids = {}
            assert cache_dir is not None
            cache_dir.mkdir(parents=True, exist_ok=True)
            def write_ids_to_disk(example, filename):
                with open(filename, 'r+b') as f:
                    mm = mmap.mmap(f.fileno(), 0)
                    start_idx = example['len_offset'] - len(example['input_ids'])
                    array_len = len(example['input_ids'])
                    arr = np.ndarray((array_len,), dtype=dtype, buffer=mm,
                                     offset=np.dtype(dtype).itemsize * start_idx)
                    arr[:] = example['input_ids']
                    mm.flush()
            for name, ds in tokenized_datasets.items():
                tokenized_datasets[name] = ds.add_column('len_offset', np.cumsum(ds['len']))
                array_len = tokenized_datasets[name][-1]['len_offset']
                filename = cache_dir / f'{name}.bin'
                # Need to create the file with this specific size first
                # https://ostechnix.com/create-files-certain-size-linux/
                subprocess.run(['truncate', '-s', str(array_len * np.dtype(dtype).itemsize),
                                str(filename)], check=True)
                tokenized_datasets[name].map(
                    write_ids_to_disk,
                    fn_kwargs={'filename': filename},
                    batched=False,
                    num_proc=max(self.num_workers, 1),
                    desc="Concatenating examples",
                )
                concat_ids[name] = np.memmap(filename, dtype=dtype, mode='r', shape=(array_len,))

        if cache_dir is not None:
            self._save_to_cache(concat_ids, tokenizer, cache_dir)
            if not self.use_shmem:
                for name in concat_ids:
                    Path(cache_dir / f'{name}.bin').unlink()
        return concat_ids, tokenizer

    def _save_to_cache(self, concat_ids, tokenizer, cache_dir):
        cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f'Saving to cache at {str(cache_dir)}')
        for k, v in concat_ids.items():
            np.save(cache_dir / f'{k}.npy', v)
        with open(cache_dir / 'tokenizer.pkl', 'wb') as f:
            pickle.dump(tokenizer, f)

    def _load_from_cache(self, cache_dir):
        assert cache_dir.is_dir()
        logger.info(f'Load from cache at {str(cache_dir)}')
        concat_ids = {split: np.load(cache_dir / f'{split}.npy', mmap_mode='r')
                      for split in ['train', 'validation', 'test']}
        with open(cache_dir / 'tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
        return concat_ids, tokenizer

    @property
    def _cache_dir_name(self):
        return f'tokenizer_name-{self.tokenizer_name}-val_ratio-{self.val_ratio}-val_split_seed-{self.val_split_seed}-add_eos-{self.add_eos}-detokenize-{self.detokenize}'

    def train_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        """ The train dataloader """
        if self.shuffle and self.fault_tolerant:
            shuffle = False
            sampler = (FaultTolerantDistributedSampler(self.dataset_train) if self.ddp
                       else RandomFaultTolerantSampler(self.dataset_train))
            # TD [2022-08-06]: Only the DDP sampler supports fast-forwarding for now
            # We assume that it's being resumed with the same number of GPUs
            if self.ddp and self.fast_forward_epochs is not None and self.fast_forward_batches is not None:
                sampler.load_state_dict({
                    'epoch': self.fast_forward_epochs,
                    'counter': self.fast_forward_batches * self.batch_size
                })
        else:
            shuffle = self.shuffle
            sampler = None
        return self._data_loader(self.dataset_train, batch_size=self.batch_size,
                                 shuffle=shuffle, sampler=sampler)

    def val_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """ The val dataloader """
        return self._data_loader(self.dataset_val, batch_size=self.batch_size_eval)

    def test_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """ The test dataloader """
        return self._data_loader(self.dataset_test, batch_size=self.batch_size_eval)

    def _data_loader(self, dataset: Dataset, batch_size: int, shuffle: bool = False,
                     sampler=None) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=1,  # Data is already in memory, we don't need many workers
            shuffle=shuffle,
            sampler=sampler,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            # persistent_workers=True
        )

    def load_state_dict(self, checkpoint):
        if self.fault_tolerant:
            self.fast_forward_epochs = checkpoint['loops']['fit_loop']['epoch_progress']['current']['completed']
            # TD [2022-08-07] ['epoch_loop.batch_progress']['total']['completed'] is 1 iteration
            # behind, so we're using the optimizer's progress. This is set correctly in seq.py.
            self.fast_forward_batches = checkpoint['loops']['fit_loop']['epoch_loop.batch_progress']['current']['completed']
        # At this point the train loader hasn't been constructed yet
