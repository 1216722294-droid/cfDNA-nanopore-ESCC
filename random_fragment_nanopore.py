#!/usr/bin/env python3
import argparse
import gzip
import numpy as np
from multiprocessing import Process, Queue
from pathlib import Path
import os
import time

class Config:
    def __init__(self, args):
        self.mean = args.mean
        self.std = args.std
        self.seed = args.seed
        self.min_len = args.min_len
        self.batch_size = args.batch_size
        self.max_queue_size = args.max_queue
        self.workers = args.workers  # 正确定义workers参数

def init_worker(config):
    global worker_rng
    process_seed = (config.seed + hash(os.getpid())) % 2**32 if config.seed else None
    worker_rng = np.random.default_rng(process_seed)

def fragment_read(header, seq, qual, config):
    total_len = len(seq)
    pos = 0
    fragments = []
    
    while pos < total_len:
        frag_len = int(worker_rng.normal(config.mean, config.std))
        frag_len = max(config.min_len, min(frag_len, total_len - pos))
        if frag_len <= 0:
            break

        fragments.append((
            f"{header}_frag{len(fragments)+1} pos={pos}-{pos+frag_len}",
            seq[pos : pos + frag_len],
            qual[pos : pos + frag_len] if qual else ""
        ))
        pos += frag_len
    
    return fragments

def reader_process(input_path, task_queue, config):
    open_func = gzip.open if Path(input_path).suffix == '.gz' else open
    with open_func(input_path, 'rt') as f:
        batch = []
        while True:
            while task_queue.qsize() > config.max_queue_size:
                time.sleep(0.1)
            
            header = f.readline().strip()
            if not header:
                break
            seq = f.readline().strip()
            f.readline()  # 跳过+
            qual = f.readline().strip()
            batch.append((header, seq, qual))
            
            if len(batch) >= config.batch_size:
                task_queue.put(batch)
                batch = []
        
        if batch:
            task_queue.put(batch)
    
    for _ in range(config.workers):  # 使用config.workers
        task_queue.put(None)

def worker_process(task_queue, result_queue, config):
    init_worker(config)
    while True:
        batch = task_queue.get()
        if batch is None:
            break
        
        for header, seq, qual in batch:
            for frag in fragment_read(header, seq, qual, config):
                result_queue.put(frag)
    
    result_queue.put(None)

def writer_process(output_path, result_queue, config):
    open_func = gzip.open if Path(output_path).suffix == '.gz' else open
    with open_func(output_path, 'wt') as f:
        active_workers = config.workers  # 使用config.workers
        while active_workers > 0:
            frag = result_queue.get()
            if frag is None:
                active_workers -= 1
                continue
            
            header, seq, qual = frag
            f.write(f"@{header}\n{seq}\n+\n{qual}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True)
    parser.add_argument('-o', '--output', required=True)
    parser.add_argument('-m', '--mean', type=int, default=150)
    parser.add_argument('-s', '--std', type=int, default=20)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--min-len', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=1000)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--max-queue', type=int, default=20)
    args = parser.parse_args()

    config = Config(args)  # 初始化配置

    task_queue = Queue(maxsize=config.max_queue_size)
    result_queue = Queue(maxsize=config.max_queue_size * 2)

    # 启动进程
    reader = Process(
        target=reader_process,
        args=(args.input, task_queue, config)
    )
    writer = Process(
        target=writer_process,
        args=(args.output, result_queue, config)
    )
    
    workers = [
        Process(
            target=worker_process,
            args=(task_queue, result_queue, config),
            daemon=True
        ) for _ in range(config.workers)  # 使用config.workers
    ]

    reader.start()
    writer.start()
    for w in workers:
        w.start()

    reader.join()
    for w in workers:
        w.join()
    writer.join()