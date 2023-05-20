import numpy as np
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

    def __init__(self) -> None:

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

        self.simmim = sm.SimMIMTransform(config)

    def __call__(self, img: NdarrayOrTensor) -> tuple[NdarrayOrTensor, NdarrayOrTensor]:
        img, mask = self.simmim(img)
        return img, mask


class SimMIMTransformWrapperd(MapTransform):
    """Dictionary wrapper for SimMIMTransformWrapper
    """

    backend = SimMIMTransformWrapper.backend

    def __init__(
            self,
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
        self.simmim = SimMIMTransformWrapper()

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
