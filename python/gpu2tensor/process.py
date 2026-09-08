"""One bounded child process, optionally reused for trusted evaluation jobs."""

import multiprocessing
import os
from pathlib import Path
import signal
import sys
import time


def child_loop(connection):
    os.setsid()
    import gc
    import torch
    from gpu2tensor.execute import evaluate
    import json

    environment = os.environ.copy()
    flags = (torch.backends.cuda.matmul.allow_tf32,
             torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
             torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction)
    for directory, backend in iter(connection.recv, None):
        directory = Path(directory)
        started = time.monotonic()
        # Restore known benchmark settings between trusted jobs. Arbitrary
        # module/native global state still requires fresh-process isolation.
        os.environ.clear()
        os.environ.update(environment)
        torch.backends.cuda.matmul.allow_tf32 = flags[0]
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = flags[1]
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = flags[2]
        with (directory / 'output/process.log').open('w') as log:
            saved = [os.dup(1), os.dup(2)]
            try:
                os.dup2(log.fileno(), 1)
                os.dup2(log.fileno(), 2)
                record = evaluate(directory, backend)
                record['process'] = {'pid': os.getpid(), 'mode': 'reused',
                                     'evaluation_seconds': time.monotonic() - started}
                (directory / 'output/record.json').write_text(json.dumps(record, indent=2, allow_nan=False))
                for name in ('gpu2tensor_candidate', 'gpu2tensor_reference', 'gpu2tensor_validator'):
                    sys.modules.pop(name, None)
                gc.collect()
                if backend == 'cuda' and torch.cuda.is_initialized():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                sys.stdout.flush()
                sys.stderr.flush()
            finally:
                for target, descriptor in enumerate(saved, start=1):
                    os.dup2(descriptor, target)
                    os.close(descriptor)
        connection.send('error' if record.get('profile_status') == 'error' else record['status'])


class Process:
    """Parent owns deadlines and recycling; one caller at a time per worker."""

    def __init__(self, max_requests=32):
        if max_requests < 1:
            raise ValueError('max_requests must be positive')
        self.max_requests = max_requests
        self.process = None
        self.connection = None
        self.requests = 0

    def close(self):
        if self.process is not None:
            alive = self.process.is_alive()
            # Reap compiler descendants even if the direct child exited first.
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                if alive:
                    self.process.kill()
            except PermissionError:
                if alive:
                    raise
            self.process.join()
            self.process.close()
            self.connection.close()
            self.process = None
            self.connection = None
            self.requests = 0

    def run(self, directory, backend, timeout):
        if self.process is None:
            context = multiprocessing.get_context('spawn')
            self.connection, child = context.Pipe()
            self.process = context.Process(target=child_loop, args=(child,))
            self.process.start()
            child.close()
        try:
            self.connection.send((str(directory), backend))
            if not self.connection.poll(timeout):
                self.close()
                return {'status': 'timeout', 'message': f'Candidate exceeded {timeout} seconds.'}
            status = self.connection.recv()
        except (EOFError, BrokenPipeError, ConnectionResetError):
            self.close()
            return {'status': 'worker_error', 'message': 'Evaluation process exited before completing the job.'}
        self.requests += 1
        # Failed jobs can leave native/runtime state unusable; always recycle.
        if status != 'ok' or self.requests >= self.max_requests:
            self.close()
        return None
