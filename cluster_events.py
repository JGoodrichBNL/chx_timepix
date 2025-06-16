import awkward as ak
import numba
import numpy as np
import pandas as pd
import pyarrow as pa

TIMESTAMP_VALUE = 1.5625*1e-9 # each raw timestamp is 1.5625 seconds
MICROSECOND = 1e-6

# We have had decent success with these values, but do not know for sure if they are optimal.
DEFAULT_CLUSTER_RADIUS = 3
DEFAULT_CLUSTER_TW_MICROSECONDS = 0.3

DEFAULT_CLUSTER_TW = int(DEFAULT_CLUSTER_TW_MICROSECONDS * MICROSECOND / TIMESTAMP_VALUE)


@numba.jit(nopython=True, cache=True)
def cluster_ak(events, radius = DEFAULT_CLUSTER_TW, tw = DEFAULT_CLUSTER_RADIUS):
    n = len(events)
    labels = np.full(n, -1, dtype=np.int64)
    cluster_id = 0

    max_time = radius * tw  # maximum time difference allowed for clustering
    radius_sq = radius ** 2

    for i in range(n):
        if labels[i] == -1:  # if event is unclustered
            labels[i] = cluster_id
            for j in range(i + 1, n):  # scan forward only
                if events[j].t - events[i].t > max_time:  # early exit based on time
                    break
                # Compute squared Euclidean distance
                dx = events[i].x - events[j].x
                dy = events[i].y - events[j].y
                dt = events[i].t - events[j].t
                distance_sq = dx**2 + dy**2 + dt**2

                if distance_sq <= radius_sq:
                    labels[j] = cluster_id
            cluster_id += 1

    return labels


if __name__ == "__main__":
    df = pd.read_parquet("example_raw_events.parquet")
    events = ak.from_arrow(pa.Table.from_pandas(df))
    labels = cluster_ak(events)
    clustered = ak.unflatten(events, ak.run_lengths(labels))
    # seems we need to do this:
    sorted_indices = np.argsort(labels)
    clustered= ak.unflatten(events[sorted_indices], ak.run_lengths(labels[sorted_indices]))
    print(len(clustered))
