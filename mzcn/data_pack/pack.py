"""Convert list of input into class:`DataPack` expected format."""

import typing

import pandas as pd
import numpy as np
#from fastHan import FastHan
import mzcn
from mzcn.engine.base_task import BaseTask


def pack(
    df: pd.DataFrame,
    task: typing.Union[str, BaseTask] = 'ranking',
) -> 'mzcn.DataPack':
    """
    Pack a :class:`DataPack` using `df`.

    The `df` must have `text_left` and `text_right` columns. Optionally,
    the `df` can have `id_left`, `id_right` to index `text_left` and
    `text_right` respectively. `id_left`, `id_right` will be automatically
    generated if not specified.

    :param df: Input :class:`pandas.DataFrame` to use.
    :param task: Could be one of `ranking`, `classification` or a
        :class:`mzcn.engine.BaseTask` instance.

    Examples::
        >>> import mzcn as mz
        >>> import pandas as pd
        >>> df = pd.DataFrame(data={'text_left': list('AABC'),
        ...                         'text_right': list('abbc'),
        ...                         'label': [0, 1, 1, 0]})
        >>> mz.pack(df, task='classification').frame()
          id_left text_left id_right text_right  label
        0     L-0         A      R-0          a      0
        1     L-0         A      R-1          b      1
        2     L-1         B      R-1          b      1
        3     L-2         C      R-2          c      0
        >>> mz.pack(df, task='ranking').frame()
          id_left text_left id_right text_right  label
        0     L-0         A      R-0          a    0.0
        1     L-0         A      R-1          b    1.0
        2     L-1         B      R-1          b    1.0
        3     L-2         C      R-2          c    0.0

    """
    if 'text_left' not in df or 'text_right' not in df:
        raise ValueError(
            'Input data frame must have `text_left` and `text_right`.')

    df = df.dropna(axis=0, how='any').reset_index(drop=True)

    # Gather IDs
    if 'id_left' not in df:
        id_left = _gen_ids(df, 'text_left', 'L-')
    else:
        id_left = df['id_left']
    if 'id_right' not in df:
        id_right = _gen_ids(df, 'text_right', 'R-')
    else:
        id_right = df['id_right']

    # Build Relation
    relation = pd.DataFrame(data={'id_left': id_left, 'id_right': id_right})
    for col in df:
        if col not in ['id_left', 'id_right', 'text_left', 'text_right']:
            relation[col] = df[col]
    if 'label' in relation:
        if task == 'classification' or isinstance(
           task, mzcn.tasks.Classification):
            relation['label'] = relation['label'].astype(int)
        elif task == 'ranking' or isinstance(task, mzcn.tasks.Ranking):
            relation['label'] = relation['label'].astype(float)
        else:
            raise ValueError(f"{task} is not a valid task.")

    # Build Left and Right
    #print('='*20)
    #print('改造后的模型')
    #model=FastHan(model_type='base')
    #print('模型加载完毕')
    #data[text_label]=change(data[text_label],model)
    left = _merge(df, id_left, 'text_left', 'id_left')
    right = _merge(df, id_right, 'text_right', 'id_right')
    return mzcn.DataPack(relation, left, right)


def change(tmp,modelT):
    tmp_list=[]
    for i in tmp:
        kkk=modelT(i,'CWS')
        tmp_list.append(str(' '.join(kkk[0])))
    return tmp_list

def _merge(data: pd.DataFrame, ids: typing.Union[list, np.array],
           text_label: str, id_label: str):
    #data[text_label]=change(data[text_label])
    left = pd.DataFrame(data={
        text_label: data[text_label], id_label: ids
    })
    left.drop_duplicates(id_label, inplace=True)
    left.set_index(id_label, inplace=True)
    return left


def _gen_ids(data: pd.DataFrame, col: str, prefix: str):
    lookup = {}
    for text in data[col].unique():
        lookup[text] = prefix + str(len(lookup))
    return data[col].map(lookup)
