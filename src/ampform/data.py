# cspell:ignore csqrt ndmin ufunc vstack
# pylint: disable=too-many-ancestors
"""Data containers for working with four-momenta.

.. seealso:: :doc:`numpy:user/basics.dispatch`
"""

from collections import abc
from typing import (
    Dict,
    ItemsView,
    Iterable,
    Iterator,
    KeysView,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
    ValuesView,
)

import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin
from numpy.lib.scimath import sqrt as csqrt

try:
    # pyright: reportMissingImports=false
    from numpy.typing import ArrayLike, DTypeLike
except ImportError:
    ArrayLike = Union[Sequence, np.ndarray]  # type: ignore
    DTypeLike = object  # type: ignore


class ScalarSequence(NDArrayOperatorsMixin, abc.Sequence):
    """`numpy.array` data container of rank 1."""

    def __init__(
        self, data: ArrayLike, dtype: Optional[DTypeLike] = None
    ) -> None:
        self.__data = np.array(data, dtype)
        if len(self.__data.shape) != 1:
            raise ValueError(
                f"{self.__class__.__name__} has to be of rank 1,"
                f" but input data is of rank {len(self.__data.shape)}"
            )

    def __array__(self, _: Optional[DTypeLike] = None) -> np.ndarray:
        return self.__data

    def __getitem__(self, i: Union[int, slice]) -> np.ndarray:  # type: ignore
        return self.__data[i]

    def __len__(self) -> int:
        return len(self.__data)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({np.array(self)}"


class ThreeMomentum(NDArrayOperatorsMixin, abc.Sequence):
    def __init__(
        self, data: ArrayLike, dtype: Optional[DTypeLike] = None
    ) -> None:
        self.__data = np.array(data, dtype=dtype, ndmin=2)
        if len(self.__data.shape) != 2:
            raise ValueError(
                f"{self.__class__.__name__} has to be of rank 2,"
                f" but this data is of rank {len(self.__data.shape)}"
            )
        if self.__data.shape[1] != 3:
            raise ValueError(
                f"{self.__class__.__name__} has to be of shape (N, 3),"
                f" but this data sample is of shape {self.__data.shape}"
            )

    def __array__(self, _: Optional[DTypeLike] = None) -> np.ndarray:
        return self.__data

    def __getitem__(  # type: ignore
        self,
        i: Union[Tuple[Union[int, slice], Union[int, slice]], int, slice],
    ) -> np.ndarray:
        return self.__data[i]

    def __len__(self) -> int:
        return len(self.__data)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({np.array(self)})"


class FourMomentumSequence(NDArrayOperatorsMixin, abc.Sequence):
    """Container for a `numpy.array` of four-momentum tuples.

    The input data has to be of shape (N, 4) and the order of the items has to
    be :math:`(E, p)` (energy first).
    """

    def __init__(self, data: ArrayLike) -> None:
        self.__data = np.array(data)
        if len(self.__data.shape) != 2:
            raise ValueError(
                f"{self.__class__.__name__} has to be of rank 2,"
                f" but this data is of rank {len(self.__data.shape)}"
            )
        if self.__data.shape[1] != 4:
            raise ValueError(
                f"{self.__class__.__name__} has to be of shape (N, 4),"
                f" but this data sample is of shape {self.__data.shape}"
            )
        if np.min(self.energy) < 0:
            raise ValueError(
                "Energy column contains entries that are less than 0."
                " Did you order the four-momentum tuples as (E, p)?"
            )

    def __array__(self, _: Optional[DTypeLike] = None) -> np.ndarray:
        return self.__data

    def __getitem__(  # type: ignore
        self,
        i: Union[Tuple[Union[int, slice], Union[int, slice]], int, slice],
    ) -> np.ndarray:
        return self.__data[i]

    def __len__(self) -> int:
        return len(self.__data)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({np.array(self)})"

    @property
    def energy(self) -> ScalarSequence:
        return ScalarSequence(self[:, 0])

    @property
    def three_momentum(self) -> ThreeMomentum:
        return ThreeMomentum(self[:, 1:])

    @property
    def p_x(self) -> ScalarSequence:
        return ScalarSequence(self[:, 1])

    @property
    def p_y(self) -> ScalarSequence:
        return ScalarSequence(self[:, 2])

    @property
    def p_z(self) -> ScalarSequence:
        return ScalarSequence(self[:, 3])

    def p_norm(self) -> ScalarSequence:
        """Norm of `.three_momentum`."""
        return ScalarSequence(np.sqrt(self.p_squared()))

    def p_squared(self) -> ScalarSequence:
        """Squared norm of `.three_momentum`."""
        return ScalarSequence(np.sum(self.three_momentum ** 2, axis=1))

    def phi(self) -> ScalarSequence:
        return ScalarSequence(np.arctan2(self.p_y, self.p_x))

    def theta(self) -> ScalarSequence:
        return ScalarSequence(np.arccos(self.p_z / self.p_norm()))

    def mass(self) -> ScalarSequence:
        mass_squared = self.mass_squared(dtype=np.float64)
        return ScalarSequence(csqrt(mass_squared))

    def mass_squared(
        self, dtype: Optional[DTypeLike] = None
    ) -> ScalarSequence:
        return ScalarSequence(
            self.energy ** 2 - self.p_norm() ** 2, dtype=dtype
        )


class MatrixSequence(NDArrayOperatorsMixin, abc.Sequence):
    """Safe data container for a sequence of 4x4-matrices."""

    def __init__(self, data: ArrayLike) -> None:
        self.__data = np.array(data)
        if len(self.__data.shape) != 3:
            raise ValueError(
                f"{self.__class__.__name__} has to be of rank 3,"
                f" but this data is of rank {len(self.__data.shape)}"
            )
        if self.__data.shape[1:] != (4, 4):
            raise ValueError(
                f"{self.__class__.__name__} has to be of shape (N, 4, 4),"
                f" but this data sample is of shape {self.__data.shape}"
            )

    def __array__(self, _: Optional[DTypeLike] = None) -> np.ndarray:
        return self.__data

    def __getitem__(self, i: Union[int, slice]) -> np.ndarray:  # type: ignore
        return self.__data[i]

    def __len__(self) -> int:
        return len(self.__data)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({np.array(self)})"

    def dot(self, vector: FourMomentumSequence) -> FourMomentumSequence:
        return FourMomentumSequence(
            np.einsum(
                "ij...,j...",
                np.transpose(self, axes=(1, 2, 0)),
                np.transpose(vector),
            )
        )


class EventCollection(abc.Mapping):
    """A mapping of state IDs to their `FourMomentumSequence` data samples.

    An `EventCollection` has to be converted to `DataSet` so that it can be
    used to evaluate a `.HelicityModel`.
    """

    def __init__(self, data: Mapping[int, ArrayLike]) -> None:
        self.__data = {i: FourMomentumSequence(v) for i, v in data.items()}
        n_events = self.n_events
        if any(map(lambda v: len(v) != n_events, self.values())):
            raise ValueError(
                f"Not all {FourMomentumSequence.__name__} items"
                f" are of length {n_events}"
            )

    def __getitem__(self, i: int) -> FourMomentumSequence:
        return self.__data[i]

    def __iter__(self) -> Iterator[int]:
        return iter(self.__data)

    def __len__(self) -> int:
        return len(self.__data)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.__data})"

    @property
    def n_events(self) -> int:
        if len(self) == 0:
            return 0
        return len(next(iter(self.values())))

    def sum(  # noqa: A003
        self, indices: Iterable[int]
    ) -> FourMomentumSequence:
        return FourMomentumSequence(sum(self.__data[i] for i in indices))  # type: ignore

    def keys(self) -> KeysView[int]:
        return self.__data.keys()

    def items(self) -> ItemsView[int, FourMomentumSequence]:
        return self.__data.items()

    def values(self) -> ValuesView[FourMomentumSequence]:
        return self.__data.values()

    def append(self, other: Mapping[int, ArrayLike]) -> None:
        if not isinstance(other, EventCollection):
            other = EventCollection(other)
        if self.n_events != 0 and set(self) != set(other):
            raise ValueError(
                f"Trying to append a momentum pool with state IDs {set(other)}"
                f" to a momentum pool with state IDs {set(self)}"
            )
        self.__data = {
            i: FourMomentumSequence(np.vstack((values, other[i])))
            for i, values in self.items()
        }

    def select_events(self, selection: Union[int, slice]) -> "EventCollection":
        return EventCollection(
            {i: values[selection] for i, values in self.items()}
        )

    def to_pandas(
        self, _: Optional[DTypeLike] = None
    ) -> Dict[Tuple[int, str], np.ndarray]:
        """Converter for the :code:`data` argument of `pandas.DataFrame`.

        The resulting `~pandas.DataFrame` has multi-columns (see
        :doc:`pandas:user_guide/advanced`) where the first column layer
        represents the state IDs and the second column layer represents
        each of the four-momentum entries (:math:`E, p_x, p_y, p_z`).
        """
        return {
            (k, label): np.transpose(v)[i]
            for k, v in self.items()
            for i, label in enumerate(["E", "px", "py", "pz"])
        }


class DataSet(abc.Mapping):
    """A mapping of variable names to their `ScalarSequence`.

    The `~.DataSet.keys` of `DataSet` represent variable names in a
    `.HelicityModel`, while its `~.DataSet.values` are inserted in their
    place.
    """

    def __init__(
        self, data: Mapping[str, ArrayLike], dtype: Optional[DTypeLike] = None
    ) -> None:
        self.__data = {
            name: ScalarSequence(v, dtype=dtype) for name, v in data.items()
        }
        if not all(map(lambda k: isinstance(k, str), self.__data)):
            raise TypeError(f"Not all keys {set(data)} are strings")
        n_events = self.n_events
        if any(map(lambda v: len(v) != n_events, self.values())):
            raise ValueError(
                f"Not all {FourMomentumSequence.__name__} items"
                f" are of length {n_events}"
            )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.__data})"

    def __getitem__(self, i: str) -> ScalarSequence:
        return self.__data[i]

    def __iter__(self) -> Iterator[str]:
        return iter(self.__data)

    def __len__(self) -> int:
        return len(self.__data)

    @property
    def n_events(self) -> int:
        if len(self) == 0:
            return 0
        return len(next(iter(self.values())))

    def keys(self) -> KeysView[str]:
        return self.__data.keys()

    def items(self) -> ItemsView[str, ScalarSequence]:
        return self.__data.items()

    def values(self) -> ValuesView[ScalarSequence]:
        return self.__data.values()

    def append(self, other: Mapping[str, ArrayLike]) -> None:
        if not isinstance(other, DataSet):
            other = DataSet(other)
        if self.n_events != 0 and set(self) != set(other):
            raise ValueError(
                f"Trying to append a data set with state IDs {set(other)}"
                f" to a data set with state IDs {set(self)}"
            )
        self.__data = {
            i: ScalarSequence(np.vstack((values, other[i])))
            for i, values in self.items()
        }

    def select_events(self, selection: Union[int, slice]) -> "DataSet":
        return DataSet({i: values[selection] for i, values in self.items()})

    def to_pandas(
        self, _: Optional[DTypeLike] = None
    ) -> Dict[str, np.ndarray]:
        """Converter for the :code:`data` argument of `pandas.DataFrame`."""
        return {k: np.array(v) for k, v in self.items()}
