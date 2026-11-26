"""E-P5 (month plan, plan of 26 Nov): pick the ImageNet-100 pretraining images from the Kaggle zip
(ambityga/imagenet100) without unpacking all 16 GB, so only the images a run will use are copied to
the RTX 5060.

Mirrors semcom.gen.data.extra_train_files: every image in the archive (train.X1..X4 and val.X parts
merge by class folder), minus any whose file name is an Imagenette image (its test split is ImageNet
training images under their ImageNet names), sorted by name, PER_CLASS evenly spaced per class. The
picked files are written as OUT/<class>/<name>, so `run_gen.py --extra-train OUT --extra-per-class
PER_CLASS` keeps exactly them.

usage: imagenet100_subset.py ZIP OUT_DIR EXCLUDE_NAMES.txt [--per-class 500]
"""
import sys
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np

argv = sys.argv[1:]
per_class = 500
if "--per-class" in argv:
    i = argv.index("--per-class")
    per_class = int(argv[i + 1])
    del argv[i:i + 2]
zpath, out, excl = Path(argv[0]), Path(argv[1]), Path(argv[2])
EXTS = {".jpg", ".jpeg", ".png", ".JPEG", ".JPG", ".PNG"}
exclude = set(excl.read_text().split())

zf = zipfile.ZipFile(zpath)
by_class: dict[str, list[zipfile.ZipInfo]] = {}
skipped = 0
for info in zf.infolist():
    p = PurePosixPath(info.filename)
    if info.is_dir() or p.suffix not in EXTS:
        continue
    if p.name in exclude:
        skipped += 1
        continue
    by_class.setdefault(p.parent.name, []).append(info)

import io  # noqa: E402
import tarfile  # noqa: E402

# OUT_DIR "-": write an uncompressed tar of <class>/<name> to stdout instead (stream it to another
# machine with `| ssh host tar x -C DIR`, so nothing lands on a full local disk).
tar = tarfile.open(fileobj=sys.stdout.buffer, mode="w|") if str(out) == "-" else None
n = 0
for c in sorted(by_class):
    fs = sorted(by_class[c], key=lambda q: PurePosixPath(q.filename).name)
    if per_class and len(fs) > per_class:
        keep = np.unique(np.linspace(0, len(fs) - 1, per_class).round().astype(int))
        fs = [fs[i] for i in keep]
    for info in fs:
        name = PurePosixPath(info.filename).name
        if tar is not None:
            data = zf.read(info)
            ti = tarfile.TarInfo(f"{c}/{name}")
            ti.size = len(data)
            tar.addfile(ti, io.BytesIO(data))
        else:
            dst = out / c / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                dst.write_bytes(zf.read(info))
        n += 1
if tar is not None:
    tar.close()
print(f"{n} images from {len(by_class)} classes into {out} ({skipped} skipped: Imagenette names)", file=sys.stderr)
