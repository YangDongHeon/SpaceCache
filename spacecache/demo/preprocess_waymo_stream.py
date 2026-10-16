"""Extract Waymo segments as a 5 Hz stream, unlabelled sweeps included.

Pointcept's preprocessing keeps only the labelled sweeps (about 2 Hz). For
the demo stream this keeps every `--stride`-th sweep of the 10 Hz LiDAR
(stride 2: 5 Hz) with the same coord / strength / pose as Pointcept.

There are no labels on most of these sweeps, so `segment.npy` is not a
ground truth: it marks the points of the four short-range LiDARs as ignored
(-1) and the top-LiDAR points with a placeholder class 0. SpaceCache's
Commit Gate only reads which points are ignored, and this is how the paper
identifies them without labels (Appendix B.1). The ground truth of the
labelled sweeps is kept in `segment_gt.npy`, which the dataset does not load.

Needs tensorflow and waymo-open-dataset (x86 Linux), like Pointcept's
preprocess_waymo.py:

    python spacecache/demo/preprocess_waymo_stream.py \\
        --raw /path/to/Waymo/validation --out data/waymo_stream \\
        --segments 8888517708810165484 4490196167747784364
"""

import argparse
import glob
import importlib.util
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np  # noqa: E402

PW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                  "pointcept", "datasets", "preprocessing", "waymo",
                  "preprocess_waymo.py")


def _pointcept_waymo():
    # by file path, so the pointcept package (and torch) is not imported
    spec = importlib.util.spec_from_file_location("preprocess_waymo", PW)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def process(tfrecord, out_root, stride):
    W = _pointcept_waymo()
    import tensorflow.compat.v1 as tf
    from waymo_open_dataset import dataset_pb2 as open_dataset
    from waymo_open_dataset.utils import frame_utils

    seq = Path(tfrecord).name.split(".")[0]
    save = Path(out_root) / "validation" / seq
    n_kept = n_lab = 0
    for i, data in enumerate(tf.data.TFRecordDataset(tfrecord)):
        if i % stride:
            continue
        frame = open_dataset.Frame()
        frame.ParseFromString(bytes(data.numpy()))
        ri, cp, _, top_pose = frame_utils.parse_range_image_and_camera_projection(
            frame)
        pts = []
        lidar = []
        for ri_index in (0, 1):
            per_lidar, _, _ = W.convert_range_image_to_point_cloud(
                frame, ri, cp, top_pose, ri_index=ri_index,
                keep_polar_features=True)
            # sorted by laser name: index 0 is TOP, 1-4 the short-range ones
            for lid, p in enumerate(per_lidar):
                pts.append(p)
                lidar.append(np.full(len(p), lid, np.int8))
        pts = np.concatenate(pts)
        lidar = np.concatenate(lidar)

        d = save / str(frame.timestamp_micros)
        os.makedirs(d, exist_ok=True)
        np.save(d / "coord.npy", pts[:, 3:6].astype(np.float32))
        np.save(d / "strength.npy", np.tanh(pts[:, 1:2]).astype(np.float32))
        np.save(d / "pose.npy",
                np.array(frame.pose.transform, np.float32).reshape(4, 4))
        np.save(d / "segment.npy", np.where(lidar == 0, 0, -1).astype(np.int32))
        if frame.lasers[0].ri_return1.segmentation_label_compressed:
            np.save(d / "segment_gt.npy",
                    (W.create_label(frame)[:, 1].reshape(-1) - 1)
                    .astype(np.int32))
            n_lab += 1
        n_kept += 1
    return f"{seq}: {n_kept} sweeps ({n_lab} labelled)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True,
                    help="directory with the validation .tfrecord files")
    ap.add_argument("--out", default="data/waymo_stream")
    ap.add_argument("--segments", nargs="+", required=True,
                    help="substrings of the segment names")
    ap.add_argument("--stride", type=int, default=2,
                    help="keep every N-th 10 Hz sweep (2: 5 Hz)")
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    files = sorted(f for f in glob.glob(os.path.join(args.raw, "*.tfrecord"))
                   if any(s in os.path.basename(f) for s in args.segments))
    if len(files) != len(args.segments):
        raise SystemExit(f"matched {len(files)} files for "
                         f"{len(args.segments)} segments: {files}")
    with ProcessPoolExecutor(args.num_workers) as pool:
        for msg in pool.map(process, files, [args.out] * len(files),
                            [args.stride] * len(files)):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
