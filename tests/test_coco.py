import numpy as np
import pytest

from pipeline.simforge.dataset.coco import decode_rle, encode_rle


def test_coco_rle_preserves_holes_edges_and_column_order():
    mask = np.array([[1, 1, 0, 0], [1, 0, 0, 1], [1, 1, 0, 1]], dtype=bool)
    assert np.array_equal(mask, decode_rle(encode_rle(mask)))
    assert encode_rle(mask)["counts"][0] == 0
    assert np.array_equal(np.zeros((3, 4), dtype=bool), decode_rle(encode_rle(np.zeros((3, 4), dtype=bool))))
    with pytest.raises(ValueError, match="span"):
        decode_rle(dict(size=[3, 4], counts=[11]))
