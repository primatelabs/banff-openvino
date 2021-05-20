# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from extensions.ops.topk import TopK
from mo.front.extractor import FrontExtractorOp
from mo.front.onnx.extractors.utils import onnx_attr, onnx_node_has_attr


class TopKExtractor(FrontExtractorOp):
    op = 'TopK'
    enabled = True

    @classmethod
    def extract(cls, node):
        """
        TopK-1 (k as attribute, required)
        TopK-10 (k as input, no sorting manipulations)
        TopK-11 (k as input, sorting manipulations through `sorted` and `largest` attrs)
        """
        attrs = {
            'axis': onnx_attr(node, 'axis', 'i', default=-1)
        }
        if onnx_node_has_attr(node, 'k'):
            attrs['k'] = onnx_attr(node, 'k', 'i')
        attrs['sort'] = 'value' if onnx_attr(node, 'sorted', 'i', default=1) else 'none'
        attrs['mode'] = 'max' if onnx_attr(node, 'largest', 'i', default=1) else 'min'

        TopK.update_node_stat(node, attrs)
        return cls.enabled
