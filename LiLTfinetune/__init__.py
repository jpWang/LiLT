from collections import OrderedDict

from transformers import CONFIG_MAPPING, MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING, MODEL_NAMES_MAPPING, TOKENIZER_MAPPING
from transformers.convert_slow_tokenizer import SLOW_TO_FAST_CONVERTERS, BertConverter, RobertaConverter, XLMRobertaConverter
from transformers.models.auto.modeling_auto import auto_class_factory

from .models.LiLTRobertaLike import (
    LiLTRobertaLikeConfig,
    LiLTRobertaLikeForRelationExtraction,
    LiLTRobertaLikeForTokenClassification,
    LiLTRobertaLikeTokenizer,
    LiLTRobertaLikeTokenizerFast,
)

CONFIG_MAPPING.update([("liltrobertalike", LiLTRobertaLikeConfig),])
MODEL_NAMES_MAPPING.update([("liltrobertalike", "LiLTRobertaLike"),])
TOKENIZER_MAPPING.update(
    [
        (LiLTRobertaLikeConfig, (LiLTRobertaLikeTokenizer, LiLTRobertaLikeTokenizerFast)),
    ]
)

with open('tag.txt', 'r') as tagf:
    TAG = tagf.read().lower()
assert TAG == 'monolingual' or TAG == 'multilingual', 'TAG is wrong. It should be monolingual or multilingual.'
if TAG == 'monolingual':
    SLOW_TO_FAST_CONVERTERS.update({"LiLTRobertaLikeTokenizer": RobertaConverter,})
elif TAG == 'multilingual':
    SLOW_TO_FAST_CONVERTERS.update({"LiLTRobertaLikeTokenizer": XLMRobertaConverter,})

MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING.update(
    [(LiLTRobertaLikeConfig, LiLTRobertaLikeForTokenClassification),]
)

MODEL_FOR_RELATION_EXTRACTION_MAPPING = OrderedDict(
    [(LiLTRobertaLikeConfig, LiLTRobertaLikeForRelationExtraction),]
)

AutoModelForTokenClassification = auto_class_factory(
    "AutoModelForTokenClassification", MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING, head_doc="token classification"
)

AutoModelForRelationExtraction = auto_class_factory(
    "AutoModelForRelationExtraction", MODEL_FOR_RELATION_EXTRACTION_MAPPING, head_doc="relation extraction"
)
