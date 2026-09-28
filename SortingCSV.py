#!/usr/bin/env python3

import os
import numpy as np

FID_DIR = "ImageOutputs"
TRACKER_FILE = "TrackerFids_IDs.csv"
OUT_FILE = "TrackerFids_IDsSorted_Modded.csv"

N_STATIONS = 18
N_FIDS_PER_STATION = 36
N_FIDS_PER_ROW = 3
MATCH_TOL = 1e-4  # mm


CSV_FORMAT = "%i, %i, %i, %i, %i, %i, %i, %8.3f, %8.3f, %8.3f, %8.3f, %8.3f, %8.3f, %8.3f, %8.3f, %8.3f"


def load_station_fids():
    real_to_mod = {}

    for st in range(N_STATIONS):
        fname = os.path.join(FID_DIR, f"FiducialsStation{st:02d}.csv")
        data = np.genfromtxt(fname, delimiter=",", names=True)

        for row in data:
            real = np.array([row["Real_X"], row["Real_Y"], row["Real_Z"]])
            mod  = np.array([row["Mod_X"],  row["Mod_Y"],  row["Mod_Z"]])

            key = tuple(np.round(real / MATCH_TOL).astype(int))

            if key in real_to_mod:
                if not np.allclose(real_to_mod[key], mod, atol=MATCH_TOL):
                    raise RuntimeError(f"Inconsistent Mod values for same Real key in {fname}")

            real_to_mod[key] = mod

    return real_to_mod


def main():
    real_to_mod = load_station_fids()

    tracker = np.genfromtxt(TRACKER_FILE, delimiter=",")
    if tracker.ndim == 1:
        tracker = tracker.reshape(1, -1)

    out_lines = []

    n_rows = tracker.shape[0]
    first_fid_col = tracker.shape[1] - 9

    n_replaced = 0
    n_unmatched = 0

    for r in range(n_rows):
        row = tracker[r].copy()

        # Replace the 3 fiducials
        for k in range(N_FIDS_PER_ROW):
            c0 = first_fid_col + 3 * k
            real_xyz = row[c0:c0 + 3]

            key = tuple(np.round(real_xyz / MATCH_TOL).astype(int))

            if key not in real_to_mod:
                n_unmatched += 1
                print(f"WARNING: no match for row {r}, fid {k}, xyz = {real_xyz}")
                continue

            row[c0:c0 + 3] = real_to_mod[key]
            n_replaced += 1

        # ---- FORMAT OUTPUT ----
        ints = row[:6].astype(int)
        floats = row[6:15]



        line = "INSERT INTO met.TrackerFiducials VALUES ("

        line = line + CSV_FORMAT % (1,
            ints[0], ints[1], ints[2], ints[3], ints[4], ints[5],
            floats[0], floats[1], floats[2],
            floats[3], floats[4], floats[5],
            floats[6], floats[7], floats[8],
        ) + ");\n"
        print(line)
        out_lines.append(line)

    with open(OUT_FILE, "w") as f:
        f.writelines(out_lines)

    print(f"Wrote: {OUT_FILE}")
    print(f"Replaced fiducials: {n_replaced}")
    print(f"Unmatched fiducials: {n_unmatched}")


if __name__ == "__main__":
    main()
