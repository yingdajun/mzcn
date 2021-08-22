from .unit import Unit
from .digit_removal import DigitRemoval
from .frequency_filter import FrequencyFilter
from .lemmatization import Lemmatization
from .lowercase import Lowercase
from .matching_histogram import MatchingHistogram
from .ngram_letter import NgramLetter
from .punc_removal import PuncRemoval
from .stateful_unit import StatefulUnit
from .stemming import Stemming
from .stop_removal import StopRemoval
from .tokenize import Tokenize
from .vocabulary import Vocabulary
from .word_hashing import WordHashing
from .character_index import CharacterIndex
from .word_exact_match import WordExactMatch
from .truncated_length import TruncatedLength
#这是我修改的部分，内置了很多中文文本预处理的方法
from .chineseStop import ChineseStopRemoval,ChineseRemoveBlack,ChineseEmotion,ChineseSimplified,IsChinese,ChineseTokenizeDemo


def list_available() -> list:
    from mzcn.utils import list_recursive_concrete_subclasses
    return list_recursive_concrete_subclasses(Unit)
