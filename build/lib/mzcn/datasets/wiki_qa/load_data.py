"""WikiQA data loader."""

import typing
import csv
from pathlib import Path

import pandas as pd

import mzcn
from mzcn.engine.base_task import BaseTask

_url = "https://download.microsoft.com/download/E/5/F/" \
       "E5FCFCEE-7005-4814-853D-DAA7C66507E0/WikiQACorpus.zip"


def load_data(
    stage: str = 'train',
    task: typing.Union[str, BaseTask] = 'ranking',
    filtered: bool = False,
    return_classes: bool = False
) -> typing.Union[mzcn.DataPack, tuple]:
    """
    Load WikiQA data.

    :param stage: One of `train`, `dev`, and `test`.
    :param task: Could be one of `ranking`, `classification` or a
        :class:`mzcn.engine.BaseTask` instance.
    :param filtered: Whether remove the questions without correct answers.
    :param return_classes: `True` to return classes for classification task,
        `False` otherwise.

    :return: A DataPack unless `task` is `classificiation` and `return_classes`
        is `True`: a tuple of `(DataPack, classes)` in that case.
    """
    if stage not in ('train', 'dev', 'test'):
        raise ValueError(f"{stage} is not a valid stage."
                         f"Must be one of `train`, `dev`, and `test`.")

    data_root = _download_data()
    file_path = data_root.joinpath(f'WikiQA-{stage}.tsv')
    data_pack = _read_data(file_path, task)
    if filtered and stage in ('dev', 'test'):
        ref_path = data_root.joinpath(f'WikiQA-{stage}.ref')
        filter_ref_path = data_root.joinpath(f'WikiQA-{stage}-filtered.ref')
        with open(filter_ref_path, mode='r') as f:
            filtered_ids = set([line.split()[0] for line in f])
        filtered_lines = []
        with open(ref_path, mode='r') as f:
            for idx, line in enumerate(f.readlines()):
                if line.split()[0] in filtered_ids:
                    filtered_lines.append(idx)
        data_pack = data_pack[filtered_lines]

    if task == 'ranking' or isinstance(task, mzcn.tasks.Ranking):
        return data_pack
    elif task == 'classification' or isinstance(
            task, mzcn.tasks.Classification):
        if return_classes:
            return data_pack, [False, True]
        else:
            return data_pack
    else:
        raise ValueError(f"{task} is not a valid task."
                         f"Must be one of `Ranking` and `Classification`.")


def _download_data():
    ref_path = mzcn.utils.get_file(
        'wikiqa', _url, extract=True,
        cache_dir=mzcn.USER_DATA_DIR,
        cache_subdir='wiki_qa'
    )
    return Path(ref_path).parent.joinpath('WikiQACorpus')


def _read_data(path, task):
    table = pd.read_csv(path, sep='\t', header=0, quoting=csv.QUOTE_NONE)
    df = pd.DataFrame({
        'text_left': table['Question'],
        'text_right': table['Sentence'],
        'id_left': table['QuestionID'],
        'id_right': table['SentenceID'],
        'label': table['Label']
    })
    return mzcn.pack(df, task)
