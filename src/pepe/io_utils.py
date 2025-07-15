import gc
import json
import logging
import os
import queue
import shutil
import threading
import time

logger = logging.getLogger("src.io_utils")


def flush_memmaps(obj):
    """Recursively flush memory maps."""
    if hasattr(obj, "flush") and callable(obj.flush):
        obj.flush()
        gc.collect()
        logger.debug("Flushed output")
    elif isinstance(obj, dict):
        for value in obj.values():
            flush_memmaps(value)


def check_disk_free_space(path, min_free_bytes):
    _, _, free = shutil.disk_usage(path)
    if free < min_free_bytes:
        raise ValueError(
            f"Not enough disk space. Required: {min_free_bytes} bytes, Available: {free} bytes"
        )
    logger.info(f"Disk space check passed. Available: {free} bytes")


class IOFlushWorker(threading.Thread):
    def __init__(self, memmap_registry, flush_bytes_limit=64 * 1024 * 1024, global_dispatcher=None):
        super().__init__()
        self.memmap_registry = memmap_registry
        self.flush_limit = flush_bytes_limit
        self.write_q = queue.Queue(maxsize=128)
        self.buffer = {}
        self.buffered_bytes = {}
        self.total_buffered = 0
        self.lock = threading.Lock()
        self.shutdown_flag = threading.Event()
        self.outstanding_enqueues = 0
        self.done_enqueuing = threading.Event()
        self.done_enqueuing.set()
        self.global_dispatcher = global_dispatcher
        self.last_checkpoint_time = time.time()
        self.checkpoint_interval = 30

    def is_range_completed(self, output_type, layer, head, offset, length):
        if self.global_dispatcher:
            return self.global_dispatcher.is_range_completed_global(output_type, layer, head, offset, length)
        return False

    def mark_range_completed(self, output_type, layer, head, offset, length):
        if self.global_dispatcher:
            self.global_dispatcher.mark_range_completed_global(output_type, layer, head, offset, length)
            now = time.time()
            if now - self.last_checkpoint_time > self.checkpoint_interval:
                self.global_dispatcher._save_global_checkpoint()
                self.last_checkpoint_time = now

    def enqueue(self, output_type, layer, head, offset, array):
        if self.is_range_completed(output_type, layer, head, offset, len(array)):
            logger.debug(
                f"[IOFlushWorker] Skipping already completed range: {output_type}, {layer}, {head}, {offset}-{offset+len(array)}"
            )
            return
        with self.lock:
            if self.outstanding_enqueues == 0:
                self.done_enqueuing.clear()
            self.outstanding_enqueues += 1
        key = (output_type, layer, head)
        try:
            while True:
                try:
                    self.write_q.put((key, offset, array), timeout=1)
                    break
                except queue.Full:
                    logger.warning("[IOFlushWorker] Write queue full, waiting to enqueue...")
                    time.sleep(0.1)
        finally:
            with self.lock:
                self.outstanding_enqueues -= 1
                if self.outstanding_enqueues == 0:
                    self.done_enqueuing.set()

    def queue_fullness(self):
        return self.write_q.qsize() / self.write_q.maxsize

    def flush_all(self):
        for key, entries in list(self.buffer.items()):
            if not entries:
                continue
            mmap = self.memmap_registry[key]
            for offset, array in entries:
                mmap[offset : offset + len(array)] = array
            mmap.flush()
            self.buffer[key] = []
            self.buffered_bytes[key] = 0
        self.total_buffered = 0

    def run(self):
        while True:
            item = self.write_q.get()
            if item is None:
                break
            key, offset, array = item
            mmap = self.memmap_registry.get(key)
            if mmap is None:
                logger.error(f"[IOFlushWorker] Unknown memmap key: {key}")
                continue
            if self.buffered_bytes.get(key, 0) + array.nbytes > self.flush_limit:
                self._flush_key(key)
            self.buffer.setdefault(key, []).append((offset, array))
            self.buffered_bytes[key] = self.buffered_bytes.get(key, 0) + array.nbytes
            self.total_buffered += array.nbytes
            if self.total_buffered >= self.flush_limit:
                self.flush_all()
            self.mark_range_completed(*key, offset, len(array))
        self.flush_all()

    def _flush_key(self, key):
        entries = self.buffer.get(key)
        if not entries:
            return
        mmap = self.memmap_registry[key]
        for offset, array in entries:
            mmap[offset : offset + len(array)] = array
        mmap.flush()
        self.total_buffered -= self.buffered_bytes.get(key, 0)
        self.buffer[key] = []
        self.buffered_bytes[key] = 0

    def stop(self, max_wait_time=60, force_shutdown=True):
        logger.info("[IOFlushWorker] Initiating shutdown...")
        self.shutdown_flag.set()
        if max_wait_time > 0:
            logger.info(f"[IOFlushWorker] Waiting up to {max_wait_time}s for pending operations...")
            completed = self.done_enqueuing.wait(timeout=max_wait_time)
            if not completed:
                with self.lock:
                    remaining = self.outstanding_enqueues
                    buffered_mb = self.total_buffered / (1024 * 1024)
                logger.warning(f"[IOFlushWorker] Timeout with {remaining} enqueues and {buffered_mb:.1f}MB buffered")
                if not force_shutdown:
                    logger.info("[IOFlushWorker] Continuing to wait since force_shutdown=False...")
                    self.done_enqueuing.wait()
                else:
                    logger.warning("[IOFlushWorker] Proceeding with forced shutdown")
        if self.global_dispatcher:
            self.global_dispatcher._save_global_checkpoint()
        while True:
            try:
                self.write_q.put(None, timeout=1)
                break
            except queue.Full:
                logger.warning("Write queue full during shutdown; retrying...")
                time.sleep(0.1)
        self.join(timeout=30)
        with self.lock:
            try:
                self.flush_all()
                if self.global_dispatcher:
                    self.global_dispatcher._save_global_checkpoint()
            except Exception as e:
                logger.error(f"[IOFlushWorker] Exception during final flush in stop(): {e}")
            remaining_in_queue = self.write_q.qsize()
            pending_buffered = sum(len(buf) for buf in self.buffer.values())
            if remaining_in_queue > 1 or pending_buffered > 0:
                logger.warning(
                    f"[IOFlushWorker] Final status: {remaining_in_queue} in queue, {pending_buffered} buffered"
                )
                if self.global_dispatcher:
                    total_ranges = sum(len(ranges) for ranges in self.global_dispatcher.global_completed_ranges.values())
                    logger.info(
                        f"[IOFlushWorker] Progress saved to global checkpoint: {len(self.global_dispatcher.global_completed_ranges)} keys, {total_ranges} ranges"
                    )
            else:
                logger.info("[IOFlushWorker] Clean shutdown - all data written")
        logger.info("[IOFlushWorker] Shutdown complete")


class MultiIODispatcher:
    def __init__(self, memmap_registry, num_workers=4, flush_bytes_limit=64 * 1024 * 1024, heavy_output_type="embeddings_unpooled", heavy_proportion=0.75, checkpoint_dir=None):
        self.num_workers = num_workers
        self.heavy_output_type = heavy_output_type
        self.checkpoint_dir = checkpoint_dir
        self.global_checkpoint_file = None
        self.global_completed_ranges = {}
        self.checkpoint_lock = threading.Lock()
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
            self.global_checkpoint_file = os.path.join(checkpoint_dir, "global_checkpoint.json")
            self._load_global_checkpoint()
        num_heavy_keys = sum(1 for key in memmap_registry if key[0] == self.heavy_output_type)
        if num_heavy_keys == 0:
            logger.warning(
                f"[MultiIODispatcher] WARNING: No keys found for heavy_output_type '{self.heavy_output_type}'. Reassigning all workers to light workload."
            )
            self.num_heavy_workers = 0
            self.num_light_workers = num_workers
        else:
            self.num_heavy_workers = max(1, int(num_workers * heavy_proportion))
            self.num_light_workers = num_workers - self.num_heavy_workers
            assert self.num_light_workers >= 1, "You need at least one light worker"
        self.workers = []
        self.heavy_workers = []
        self.light_workers = []
        sharded_registries = [{} for _ in range(num_workers)]
        for key, mmap in memmap_registry.items():
            output_type = key[0]
            if output_type == self.heavy_output_type and self.num_heavy_workers > 0:
                shard_id = hash(key) % self.num_heavy_workers
            else:
                shard_id = self.num_heavy_workers + (hash(key) % self.num_light_workers)
            sharded_registries[shard_id][key] = mmap
        for i, reg in enumerate(sharded_registries):
            logger.info(f"[MultiIODispatcher] Worker {i} assigned {len(reg)} keys")
        for i in range(num_workers):
            worker = IOFlushWorker(memmap_registry=sharded_registries[i], flush_bytes_limit=flush_bytes_limit, global_dispatcher=self)
            worker.start()
            self.workers.append(worker)
        self.heavy_workers = self.workers[: self.num_heavy_workers] if self.num_heavy_workers > 0 else []
        self.light_workers = self.workers[self.num_heavy_workers :]

    def queue_fullness(self):
        return max(worker.queue_fullness() for worker in self.workers)

    def enqueue(self, output_type, layer, head, offset, array):
        key = (output_type, layer, head)
        if output_type == self.heavy_output_type:
            worker_id = hash(key) % self.num_heavy_workers
            self.heavy_workers[worker_id].enqueue(output_type, layer, head, offset, array)
        else:
            worker_id = hash(key) % self.num_light_workers
            self.light_workers[worker_id].enqueue(output_type, layer, head, offset, array)

    def stop(self, max_wait_time=60, force_shutdown=True):
        logger.info(f"[MultiIODispatcher] Stopping {len(self.workers)} workers...")
        for i, worker in enumerate(self.workers):
            logger.info(f"[MultiIODispatcher] Stopping worker {i}...")
            worker.stop(max_wait_time=max_wait_time, force_shutdown=force_shutdown)
        logger.info("[MultiIODispatcher] All workers stopped")

    def get_resume_info(self):
        total_completed_ranges = sum(len(ranges) for ranges in self.global_completed_ranges.values())
        if total_completed_ranges > 0:
            logger.info(f"[MultiIODispatcher] Resume info:")
            for key, ranges in self.global_completed_ranges.items():
                total_bytes = sum(end - start for start, end in ranges)
                logger.info(f"  {key}: {len(ranges)} ranges, {total_bytes / (1024*1024):.1f}MB")
        return {
            "num_workers": len(self.workers),
            "completed_ranges": sum(len(r) for r in self.global_completed_ranges.values()),
        }

    def _load_global_checkpoint(self):
        if not self.global_checkpoint_file:
            return
        try:
            with open(self.global_checkpoint_file, "r") as f:
                data = json.load(f)
                for key_str, ranges in data.get("completed_ranges", {}).items():
                    parts = key_str.split("|")
                    if len(parts) == 3:
                        output_type, layer, head = parts
                        layer = int(layer) if layer != "None" else None
                        head = int(head) if head != "None" else None
                        key = (output_type, layer, head)
                        self.global_completed_ranges[key] = set(tuple(r) for r in ranges)
                total_ranges = sum(len(ranges) for ranges in self.global_completed_ranges.values())
                logger.info(
                    f"[MultiIODispatcher] Loaded global checkpoint: {len(self.global_completed_ranges)} keys, {total_ranges} ranges"
                )
                if total_ranges > 0:
                    logger.info("[MultiIODispatcher] Resume info:")
                    for key, ranges in self.global_completed_ranges.items():
                        total_bytes = sum(end - start for start, end in ranges)
                        logger.info(
                            f"  {key}: {len(ranges)} ranges, {total_bytes / (1024*1024):.1f}MB"
                        )
        except Exception as e:
            logger.error(f"[MultiIODispatcher] Failed to load global checkpoint: {e}")
            self.global_completed_ranges = {}

    def _save_global_checkpoint(self):
        if not self.global_checkpoint_file:
            return
        with self.checkpoint_lock:
            try:
                data = {
                    "completed_ranges": {
                        f"{key[0]}|{key[1]}|{key[2]}": list(ranges)
                        for key, ranges in self.global_completed_ranges.items()
                    },
                    "timestamp": time.time(),
                    "num_workers_used": self.num_workers,
                }
                temp_file = self.global_checkpoint_file + ".tmp"
                with open(temp_file, "w") as f:
                    json.dump(data, f, indent=2)
                os.rename(temp_file, self.global_checkpoint_file)
                total_ranges = sum(len(ranges) for ranges in self.global_completed_ranges.values())
                logger.info(
                    f"[MultiIODispatcher] Saved global checkpoint: {len(self.global_completed_ranges)} keys, {total_ranges} ranges"
                )
            except Exception as e:
                logger.error(f"[MultiIODispatcher] Failed to save global checkpoint: {e}")

    def is_range_completed_global(self, output_type, layer, head, offset, length):
        key = (output_type, layer, head)
        if key not in self.global_completed_ranges:
            return False
        end_offset = offset + length
        completed_set = self.global_completed_ranges[key]
        for start, end in completed_set:
            if start <= offset and end_offset <= end:
                return True
        return False

    def mark_range_completed_global(self, output_type, layer, head, offset, length):
        key = (output_type, layer, head)
        with self.checkpoint_lock:
            if key not in self.global_completed_ranges:
                self.global_completed_ranges[key] = set()
            self.global_completed_ranges[key].add((offset, offset + length))
            self._merge_ranges_global(key)

    def _merge_ranges_global(self, key):
        if key not in self.global_completed_ranges:
            return
        ranges = sorted(self.global_completed_ranges[key])
        merged = []
        for start, end in ranges:
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        self.global_completed_ranges[key] = set(merged)
