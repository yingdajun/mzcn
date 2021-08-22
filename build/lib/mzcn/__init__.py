"""
由于这个包的本质是对matchzoo-py这个包的二次拓展，所以很多自动下载的部分仍然是matchzoo方面的修改
"""
from pathlib import Path

USER_DIR = Path.expanduser(Path('~')).joinpath('.matchzoo')
if not USER_DIR.exists():
    USER_DIR.mkdir()
USER_DATA_DIR = USER_DIR.joinpath('datasets')
if not USER_DATA_DIR.exists():
    USER_DATA_DIR.mkdir()
USER_TUNED_MODELS_DIR = USER_DIR.joinpath('tuned_models')

from .version import __version__

from .data_pack import DataPack
from .data_pack import pack
from .data_pack import load_data_pack

from . import preprocessors
from . import dataloader

from .preprocessors.chain_transform import chain_transform

from . import auto
from . import tasks
from . import metrics
from . import losses
from . import engine
from . import models
from . import trainers
from . import embedding
from . import datasets
from . import modules

from .engine import hyper_spaces
from .engine.base_preprocessor import load_preprocessor
from .engine.param import Param
from .engine.param_table import ParamTable

from .embedding.embedding import Embedding

from .preprocessors.build_unit_from_data_pack import build_unit_from_data_pack
from .preprocessors.build_vocab_unit import build_vocab_unit
