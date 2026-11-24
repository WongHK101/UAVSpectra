"""Exact set-union scheduling for the unchanged native-Mesh scoring stream.

The frozen scorer still owns coordinates, clipping, triangle areas, IID random
samples, voxel encoding/centres, and nearest-neighbour metrics. Only the order
of associative, exact uint64 set unions changes; the final IDs are sorted.
"""
import math
import time

import numpy as np


class ExactUnion:
    def __init__(self):
        self.levels = []

    def add(self, ids):
        carry = np.unique(np.asarray(ids, dtype=np.uint64))
        if not len(carry):
            return
        level = 0
        while level < len(self.levels) and self.levels[level] is not None:
            carry = np.union1d(self.levels[level], carry)
            self.levels[level] = None
            level += 1
        if level == len(self.levels):
            self.levels.append(carry)
        else:
            self.levels[level] = carry

    def finish(self):
        ids = np.empty(0, dtype=np.uint64)
        for block in self.levels:
            if block is not None:
                ids = np.union1d(ids, block)
        return ids


def install(scorer):
    """Replace just voxel_sample on an independently imported frozen scorer."""
    def voxel_sample(triangles, core, density, seed):
        total = math.ceil(float(scorer.triangle_areas(triangles).sum()) * density)
        started = time.monotonic()
        print(f'sampling_target={total} density={density} seed={seed}', flush=True)
        union = ExactUnion()
        pending = []
        count = 0
        for xyz in scorer.sample_batches(triangles, density, seed):
            pending.append(core.voxel_batch_ids(xyz, .05, np.zeros(3, dtype=np.float64)))
            count += len(xyz)
            if len(pending) == 16:
                union.add(np.concatenate(pending))
                pending.clear()
                elapsed = time.monotonic() - started
                rate = count / max(elapsed, .001)
                print(f'sampled={count}/{total} samples_per_second={rate:.1f} '
                      f'sampling_eta_seconds={(total-count)/rate:.1f} '
                      f'union_blocks={sum(b is not None for b in union.levels)}', flush=True)
        if pending:
            union.add(np.concatenate(pending))
        ids = union.finish()
        if count != total:
            raise ValueError('frozen sample stream length mismatch')
        print(f'sampling_complete={count} unique_voxels={len(ids)} '
              f'sampling_seconds={time.monotonic()-started:.3f}', flush=True)
        return core.voxel_centers_local(ids, .05), count
    scorer.voxel_sample = voxel_sample
