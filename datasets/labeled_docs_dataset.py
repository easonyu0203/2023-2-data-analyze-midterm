from typing import Protocol, Tuple, List, Any

from datasets.document import Document


class ILabeledDataset(Protocol):
    """
    A labeled dataset, where x can be a document or list of keywords or vector, and y is a float value.
    this return (x, y) where x is a document and y is a float value.
    """

    features: list[Any]
    labels: list[float]

    def __getitem__(self, index: int) -> Tuple[Any, float]:
        ...

    def __len__(self) -> int:
        ...

    def __next__(self) -> Tuple[Any, float]:
        ...


class LabelDataset(ILabeledDataset):
    """
    A labeled dataset, where x can be a document or list of keywords or vector, and y is a float value.
    this return (x, y) where x is a document and y is a float value.
    """

    def __init__(self, features: list[Any], labels: list[float]):
        assert len(features) == len(labels)
        self.features = features
        self.labels = labels

    def __getitem__(self, index) -> Tuple[Any, float]:
        """this return (x, y) where x is a document and y is a float value."""
        return self.features[index], self.labels[index]

    def __len__(self) -> int:
        return len(self.features)

    def __iter__(self):
        self.__next_index = 0
        return self

    def __next__(self):
        if self.__next_index < len(self.features):
            self.__next_index += 1
            return self.features[self.__next_index - 1], self.labels[self.__next_index - 1]
        else:
            raise StopIteration

    def __repr__(self):
        return f"LabelDocsDataset({self.features}, {self.labels})"

    def __str__(self):
        return f"LabelDocsDataset({self.features}, {self.labels})"
