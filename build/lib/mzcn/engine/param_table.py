"""Parameters table class."""

import typing
import pandas as pd
import collections.abc

from mzcn.engine.param import Param
from mzcn.engine import hyper_spaces


class ParamTable(object):
    """
    Parameter table class.

    Example:

        >>> params = ParamTable()
        >>> params.add(Param('ham', 'Parma Ham'))
        >>> params.add(Param('egg', 'Over Easy'))
        >>> params['ham']
        'Parma Ham'
        >>> params['egg']
        'Over Easy'
        >>> print(params)
        ham                           Parma Ham
        egg                           Over Easy
        >>> params.add(Param('egg', 'Sunny side Up'))
        Traceback (most recent call last):
            ...
        ValueError: Parameter named egg already exists.
        To re-assign parameter egg value, use `params["egg"] = value` instead.
    """

    def __init__(self):
        """Parameter table constrctor."""
        self._params = {}

    def add(self, param: Param):
        """:param param: parameter to add."""
        if not isinstance(param, Param):
            raise TypeError("Only accepts a Param instance.")
        if param.name in self._params:
            msg = f"Parameter named {param.name} already exists.\n" \
                f"To re-assign parameter {param.name} value, " \
                f"use `params[\"{param.name}\"] = value` instead."
            raise ValueError(msg)
        self._params[param.name] = param

    def get(self, key) -> Param:
        """:return: The parameter in the table named `key`."""
        return self._params[key]

    def set(self, key, param: Param):
        """Set `key` to parameter `param`."""
        if not isinstance(param, Param):
            raise ValueError("Only accepts a Param instance.")
        self._params[key] = param

    @property
    def hyper_space(self) -> dict:
        """:return: Hyper space of the table, a valid `hyperopt` graph."""
        full_space = {}
        for param in self:
            if param.hyper_space is not None:
                param_space = param.hyper_space
                if isinstance(param_space, hyper_spaces.HyperoptProxy):
                    param_space = param_space.convert(param.name)
                full_space[param.name] = param_space
        return full_space

    def to_frame(self) -> pd.DataFrame:
        """
        Convert the parameter table into a pandas data frame.

        :return: A `pandas.DataFrame`.

        Example:
            >>> import mzcn as mz
            >>> table = mz.ParamTable()
            >>> table.add(mz.Param(name='x', value=10, desc='my x'))
            >>> table.add(mz.Param(name='y', value=20, desc='my y'))
            >>> table.to_frame()
              Name Description  Value Hyper-Space
            0    x        my x     10        None
            1    y        my y     20        None

        """
        df = pd.DataFrame(data={
            'Name': [p.name for p in self],
            'Description': [p.desc for p in self],
            'Value': [p.value for p in self],
            'Hyper-Space': [p.hyper_space for p in self]
        }, columns=['Name', 'Description', 'Value', 'Hyper-Space'])
        return df

    def __getitem__(self, key: str) -> typing.Any:
        """:return: The value of the parameter in the table named `key`."""
        return self._params[key].value

    def __setitem__(self, key: str, value: typing.Any):
        """
        Set the value of the parameter named `key`.

        :param key: Name of the parameter.
        :param value: New value of the parameter to set.
        """
        self._params[key].value = value

    def __str__(self):
        """:return: Pretty formatted parameter table."""
        return '\n'.join(param.name.ljust(30) + str(param.value)
                         for param in self._params.values())

    def __iter__(self) -> typing.Iterator:
        """:return: A iterator that iterates over all parameter instances."""
        yield from self._params.values()

    def completed(self, exclude: typing.Optional[list] = None) -> bool:
        """
        Check if all params are filled.

        :param exclude: List of names of parameters that was excluded
            from being computed.

        :return: `True` if all params are filled, `False` otherwise.

        Example:

            >>> import mzcn
            >>> model = mzcn.models.DenseBaseline()
            >>> model.params.completed(
            ...     exclude=['task', 'out_activation_func', 'embedding',
            ...              'embedding_input_dim', 'embedding_output_dim']
            ... )
            True

        """
        return all(param for param in self if param.name not in exclude)

    def keys(self) -> collections.abc.KeysView:
        """:return: Parameter table keys."""
        return self._params.keys()

    def __contains__(self, item):
        """:return: `True` if parameter in parameters."""
        return item in self._params

    def update(self, other: dict):
        """
        Update `self`.

        Update `self` with the key/value pairs from other, overwriting
        existing keys. Notice that this does not add new keys to `self`.

        This method is usually used by models to obtain useful information
        from a preprocessor's context.

        :param other: The dictionary used update.

        Example:
            >>> import mzcn as mz
            >>> model = mz.models.DenseBaseline()
            >>> prpr = model.get_default_preprocessor()
            >>> _ = prpr.fit(mz.datasets.toy.load_data(), verbose=0)
            >>> model.params.update(prpr.context)

        """
        for key in other:
            if key in self:
                self[key] = other[key]
