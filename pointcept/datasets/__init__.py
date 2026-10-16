# open3d is optional: several dataset modules import it at module scope, but
# only get_normals() (the *ImagePointDataset path) calls it. Without open3d a
# stub module is registered whose public attributes raise ImportError on use.
import sys as _sys
import types as _types

try:
    import open3d as _open3d  # noqa: F401
except ImportError:

    class _MissingOpen3D(_types.ModuleType):
        def __getattr__(self, name):
            # private/dunder lookups must raise AttributeError: code that
            # scans sys.modules with hasattr() only catches AttributeError
            if name.startswith("_"):
                raise AttributeError(name)
            raise ImportError(
                "open3d is not installed, but this code path needs it "
                f"(tried to access `open3d.{name}`). It is only required by "
                "the *ImagePointDataset / get_normals() path; install it with "
                "`pip install open3d`."
            )

    _sys.modules["open3d"] = _MissingOpen3D("open3d")
else:
    del _open3d

del _sys, _types

from .defaults import (
    DefaultDataset,
    DefaultImagePointDataset,
    DefaultMultiViewImagePointDataset,
    ConcatDataset,
)
from .builder import build_dataset
from .utils import point_collate_fn, collate_fn

# indoor scene
from .scannet import (
    ScanNetDataset,
    ScanNet200Dataset,
    ScanNetVideoDataset,
)

# outdoor scene
from .semantic_kitti import SemanticKITTIDataset, SemanticKITTIImagePointDataset
from .nuscenes import NuScenesDataset, NuScenesImagePointDataset
from .waymo import WaymoDataset

# dataloader
from .dataloader import MultiDatasetDataloader
