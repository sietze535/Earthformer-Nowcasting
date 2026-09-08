"""
Virtual concatenation of multiple zarr stores along the time axis.

Provides proxy objects (MultiZarrRoot, MultiZarrGroup, ConcatenatedArray)
that make N per-year zarr stores look like a single store to the dataloader.
No data is copied — reads are routed to the correct year's store at access time.
"""

import bisect
import numpy as np
import zarr


class ConcatenatedArray:
    """Lazy concatenation of multiple zarr arrays along axis 0 (time)."""

    def __init__(self, arrays: list[zarr.Array]):
        self._arrays = arrays
        self._lengths = [a.shape[0] for a in arrays]
        self._cumsum = np.cumsum(self._lengths)
        self.shape = (sum(self._lengths),) + arrays[0].shape[1:]
        self.dtype = arrays[0].dtype
        self.chunks = arrays[0].chunks

    def _resolve(self, idx: int) -> tuple[int, int]:
        """Map global time index to (array_index, local_index)."""
        arr_idx = bisect.bisect_right(self._cumsum, idx)
        local_idx = idx - (self._cumsum[arr_idx - 1] if arr_idx > 0 else 0)
        return arr_idx, local_idx

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)):
            arr_idx, local_idx = self._resolve(key)
            return self._arrays[arr_idx][local_idx]

        if not isinstance(key, tuple):
            key = (key,)

        t_key = key[0]
        rest = key[1:]

        if isinstance(t_key, slice):
            start = t_key.start or 0
            stop = t_key.stop if t_key.stop is not None else self.shape[0]

            start_arr, local_start = self._resolve(start)
            end_arr, _ = self._resolve(min(stop - 1, self.shape[0] - 1))

            if start_arr == end_arr:
                local_stop = local_start + (stop - start)
                return self._arrays[start_arr][(slice(local_start, local_stop),) + rest]

            # Spans multiple arrays — concatenate chunks
            chunks = []
            for ai in range(start_arr, end_arr + 1):
                offset = self._cumsum[ai - 1] if ai > 0 else 0
                ls = max(start - offset, 0)
                le = min(stop - offset, self._lengths[ai])
                chunks.append(self._arrays[ai][(slice(ls, le),) + rest])
            return np.concatenate(chunks, axis=0)

        if isinstance(t_key, np.ndarray):
            arr_idx, _ = self._resolve(int(t_key[0]))
            offset = self._cumsum[arr_idx - 1] if arr_idx > 0 else 0
            local_indices = t_key - offset
            return self._arrays[arr_idx][(local_indices,) + rest]

        if isinstance(t_key, (int, np.integer)):
            arr_idx, local_idx = self._resolve(int(t_key))
            return self._arrays[arr_idx][(local_idx,) + rest]

        raise TypeError(f"Unsupported index type: {type(t_key)}")


class MultiZarrGroup:
    """Proxy for a zarr group that concatenates arrays from multiple stores."""

    def __init__(self, groups: list[zarr.Group]):
        self._groups = groups
        first_attrs = dict(groups[0].attrs)
        last_attrs = dict(groups[-1].attrs)
        self.attrs = {
            **first_attrs,
            "start_time": first_attrs.get("start_time"),
            "end_time": last_attrs.get("end_time"),
        }

    def __contains__(self, key):
        return key in self._groups[0]

    def __getitem__(self, key):
        arrays = [g[key] for g in self._groups]
        if all(isinstance(a, zarr.Array) for a in arrays):
            return ConcatenatedArray(arrays)
        raise KeyError(f"'{key}' is not an array in all groups")


class MultiZarrRoot:
    """
    Proxy for a zarr root that wraps multiple per-year stores.

    Usage:
        roots = [zarr.open_group(p, mode='r') for p in paths]
        multi = MultiZarrRoot(roots)
        # Now use multi exactly like a single zarr.Group:
        data = multi["radar/rtcor"][0:100, :, :]
        start = multi["radar"].attrs["start_time"]
    """

    def __init__(self, roots: list[zarr.Group]):
        self._roots = roots

    def __contains__(self, key):
        return key in self._roots[0]

    def __getitem__(self, key):
        if "/" in key:
            group_name, array_name = key.split("/", 1)
            group = self[group_name]
            return group[array_name]

        groups = []
        for root in self._roots:
            if key not in root:
                raise KeyError(f"Group '{key}' not found in all zarr stores")
            groups.append(root[key])
        return MultiZarrGroup(groups)
