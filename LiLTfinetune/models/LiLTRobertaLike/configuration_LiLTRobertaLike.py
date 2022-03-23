# coding=utf-8
from collections import OrderedDict
from typing import Any, List, Mapping, Optional

from transformers import PretrainedConfig, PreTrainedTokenizer, TensorType

from transformers.utils import logging
from transformers import RobertaConfig, XLMRobertaConfig

logger = logging.get_logger(__name__)

with open('tag.txt', 'r') as tagf:
    TAG = tagf.read().lower()
assert TAG == 'monolingual' or TAG == 'multilingual', 'TAG is wrong. It should be monolingual or multilingual.'

if TAG == 'monolingual':
    class LiLTRobertaLikeConfig(RobertaConfig):
        model_type = "liltrobertalike"

        def __init__(
            self,
            channel_shrink_ratio=4,
            max_2d_position_embeddings=1024,
            **kwargs
        ):
            super().__init__(
                **kwargs,
            )
            self.channel_shrink_ratio = channel_shrink_ratio
            self.max_2d_position_embeddings = max_2d_position_embeddings

elif TAG == 'multilingual':
    class LiLTRobertaLikeConfig(XLMRobertaConfig):
        model_type = "liltrobertalike"

        def __init__(
            self,
            channel_shrink_ratio=4,
            max_2d_position_embeddings=1024,
            **kwargs
        ):
            super().__init__(
                **kwargs,
            )
            self.channel_shrink_ratio = channel_shrink_ratio
            self.max_2d_position_embeddings = max_2d_position_embeddings
