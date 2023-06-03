import numpy as np
from skimage.transform import resize
from typing import Callable, Dict, Hashable, Mapping, Optional, Sequence, Tuple, Union
import simmim.data.data_simmim as sm

from monai.transforms import MapTransform
from monai.config import NdarrayOrTensor, DtypeLike, KeysCollection
from monai.transforms.transform import Transform
from monai.utils import convert_to_tensor, ensure_tuple, ensure_tuple_rep
from monai.utils.enums import TransformBackends, PostFix

from argparse import Namespace

DEFAULT_POST_FIX = PostFix.meta()


class SimMIMTransformWrapper(Transform):
    """
    MONAI API Compatible Wrapper for SimMIMTransformWrapper

    Args:
        subtrahend: the amount to subtract by (usually the mean).
        divisor: the amount to divide by (usually the standard deviation).
        nonzero: whether only normalize non-zero values.
        channel_wise: if True, calculate on each channel separately, otherwise, calculate on
            the entire image directly. default to False.
        dtype: output data type, if None, same as input image. defaults to float32.
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(self, roi_size, phi_x, m) -> None:

        config = Namespace(**{
            "DATA_IMG_SIZE": None,
            "MODEL": Namespace(**{
                "TYPE": "vit",
                "VIT": Namespace(**{
                    "PATCH_SIZE": 16
                })
            }),
            "DATA": Namespace(**{
                "IMG_SIZE": 96,
                "MASK_PATCH_SIZE": 32,
                "MASK_RATIO": 0.6
            })
        })

        self.roi_x = roi_size[0]
        self.roi_y = roi_size[1]
        self.roi_z = roi_size[2]
        self.phi_x = phi_x
        self.m = m
        self.simmim = sm.SimMIMTransform(config)

    def __call__(self, img: NdarrayOrTensor) -> tuple[NdarrayOrTensor, NdarrayOrTensor]:
        mask_points = (np.random.rand(self.roi_x) < self.phi_x) * 1

        for i, x in enumerate(mask_points):
            if x == 1:
                for j in range(i, i - self.m, -1):
                    mask_points[j] = 1

        # get groups of all consecutive elements with one value
        mask_indices = np.where(abs(np.diff(mask_points)) == 1)[0]+1
        mask_groups = np.split(mask_points, mask_indices)
        mask_indices = np.array([0, *mask_indices])

        g_masks = []

        for i, g in zip(mask_indices, mask_groups):
            g_val = g[0]
            g_size_x = len(g)
            g_size_y = img.shape[2]
            g_size_z = img.shape[3]
            g_slice = img[0, i, :, :]

            if g_val == 1:
                _, g_mask = self.simmim(g_slice)
                g_mask = resize(g_mask, (1, g_size_y, g_size_z), order=0)
                g_mask = np.repeat(g_mask, g_size_x, axis=0)
            else:
                g_mask = np.ones((g_size_x, g_size_y, g_size_z))

            g_masks.append(g_mask)

        mask = np.concatenate(g_masks)
        return img, mask

class SimMIMTransformWrapperd(MapTransform):
    """Dictionary wrapper for SimMIMTransformWrapper
    """

    backend = SimMIMTransformWrapper.backend

    def __init__(
            self,
            roi_size, phi_x, m,
            keys: KeysCollection,
            factor_key: Optional[str] = None,
            meta_keys: Optional[KeysCollection] = None,
            meta_key_postfix: str = DEFAULT_POST_FIX,
            allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.factor_key = ensure_tuple_rep(factor_key, len(self.keys))
        self.meta_keys = ensure_tuple_rep(None, len(self.keys)) if meta_keys is None else ensure_tuple(meta_keys)
        if len(self.keys) != len(self.meta_keys):
            raise ValueError("meta_keys should have the same length as keys.")
        self.meta_key_postfix = ensure_tuple_rep(meta_key_postfix, len(self.keys))
        self.simmim = SimMIMTransformWrapper(roi_size, phi_x, m)

    def __call__(self, data) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key, factor_key, meta_key, meta_key_postfix in self.key_iterator(
            d, self.factor_key, self.meta_keys, self.meta_key_postfix
        ):
            meta_key = meta_key or f"{key}_{meta_key_postfix}"
            img, mask = self.simmim(d[key])
            d[key] = img
            d["mask"] = mask

        return d
