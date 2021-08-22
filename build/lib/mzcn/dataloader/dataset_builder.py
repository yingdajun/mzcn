import mzcn as mz
from mzcn.dataloader import Dataset


class DatasetBuilder(object):
    """
    Dataset Bulider. In essense a wrapped partial function.

    Example:
        >>> import mzcn as mz
        >>> builder = mz.dataloader.DatasetBuilder(
        ...     mode='point'
        ... )
        >>> data = mz.datasets.toy.load_data()
        >>> gen = builder.build(data)
        >>> type(gen)
        <class 'mzcn.dataloader.dataset.Dataset'>

    """

    def __init__(self, **kwargs):
        """Init."""
        self._kwargs = kwargs

    def build(self, data_pack, **kwargs) -> Dataset:
        """
        Build a Dataset.

        :param data_pack: DataPack to build upon.
        :param kwargs: Additional keyword arguments to override the keyword
            arguments passed in `__init__`.
        """
        return mz.dataloader.Dataset(
            data_pack, **{**self._kwargs, **kwargs}
        )
