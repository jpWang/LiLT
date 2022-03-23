# coding=utf-8

from transformers import RobertaTokenizer, XLMRobertaTokenizer
from transformers.utils import logging

logger = logging.get_logger(__name__)

SPIECE_UNDERLINE = "▁"

VOCAB_FILES_NAMES = {"vocab_file": "sentencepiece.bpe.model"}

with open('tag.txt', 'r') as tagf:
    TAG = tagf.read().lower()
assert TAG == 'monolingual' or TAG == 'multilingual', 'TAG is wrong. It should be monolingual or multilingual.'

if TAG == 'monolingual':
    class LiLTRobertaLikeTokenizer(RobertaTokenizer):
        vocab_files_names = VOCAB_FILES_NAMES
        max_model_input_sizes = {"lilt-roberta-base": 512,}
        model_input_names = ["input_ids", "attention_mask"]

        def __init__(self, model_max_length=512, **kwargs):
            super().__init__(model_max_length=model_max_length, **kwargs)

elif TAG == 'multilingual':
    class LiLTRobertaLikeTokenizer(XLMRobertaTokenizer):
        vocab_files_names = VOCAB_FILES_NAMES
        max_model_input_sizes = {"lilt-infoxlm-base": 512,}
        model_input_names = ["input_ids", "attention_mask"]

        def __init__(self, model_max_length=512, **kwargs):
            super().__init__(model_max_length=model_max_length, **kwargs)
