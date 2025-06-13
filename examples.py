import pandas as pd
import numba
import awkward as ak
import numpy as np


df = pd.DataFrame({
    "x": [1, 2, 10, 30, 31, 31],
    "y": [1, 1, 10, 30, 30, 32],
    "t": [0, 1, 15, 23, 23, 24],
    "ToT": [300, 200, 500, 150, 175, 150],
    "cluster_id": [1, 1, 2, 3, 3, 3],
})
a = ak.Array([
    [{"x": 1, "y": 1, "t": 0, "ToT": 300}, {"x": 2, "y": 1, "t": 1, "ToT": 200}],
    [{"x": 10, "y": 10, "t": 15, "ToT": 500}],
    [{"x": 30, "y": 30, "t": 23, "ToT": 150}, {"x": 31, "y": 30, "t": 23, "ToT": 175}, {"x": 31, "y": 32, "t": 24, "ToT": 150}]
])

# Awkward encodes clusters more compactly by recording bin edges instead of
# labeling every event.
a.nbytes
df.values.nbytes

# number of events per centroid
n = ak.num(a['t'], axis=1)
# centroids
xc = ak.sum(a['x'] * a['ToT'], axis=1) / ak.sum(a['ToT'], 1)
yc = ak.sum(a['y'] * a['ToT'], axis=1) / ak.sum(a['ToT'], 1)
# ToT stats per centroid
ToT_max = ak.max(a['ToT'], axis=1)
ToT_sum = ak.sum(a['ToT'], axis=1)
# timestamp of the largest ToT per centroid
t = ak.flatten(a['t'][ak.argmax(a['ToT'], axis=1, keepdims=True)])


@numba.jit(nopython=True, cache=True)
def centroid_clusters(
    cluster_arr: np.ndarray, events: np.ndarray
) -> tuple[np.ndarray]:  

    num_clusters = cluster_arr.shape[0]
    max_cluster = cluster_arr.shape[1]
    t = np.zeros(num_clusters, dtype="uint64")
    xc = np.zeros(num_clusters, dtype="float32")
    yc = np.zeros(num_clusters, dtype="float32")
    ToT_max = np.zeros(num_clusters, dtype="uint32")
    ToT_sum = np.zeros(num_clusters, dtype="uint32")
    n = np.zeros(num_clusters, dtype="ubyte")

    for cluster_id in range(num_clusters):
        _ToT_max = np.ushort(0)
        for event_num in range(max_cluster):
            event = cluster_arr[cluster_id, event_num]
            if event > -1:  # if we have an event here
                if events[event, 2] > _ToT_max:  # find the max ToT, assign, use that time
                    _ToT_max = events[event, 2]
                    t[cluster_id] = events[event, 3]
                    ToT_max[cluster_id] = _ToT_max
                xc[cluster_id] += events[event, 0] * events[event, 2]  # x and y centroids by time over threshold
                yc[cluster_id] += events[event, 1] * events[event, 2]
                ToT_sum[cluster_id] += events[event, 2]  # calcuate sum
                n[cluster_id] += np.ubyte(1)  # number of events in cluster
            else:
                break
        xc[cluster_id] /= ToT_sum[cluster_id]  # normalize
        yc[cluster_id] /= ToT_sum[cluster_id]

    return t, xc, yc, ToT_max, ToT_sum, n

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


events = ak.flatten(a)
labels = cluster_ak(events, 3, 0.5)
clustered_events = ak.unflatten(events, ak.run_lengths(labels))
