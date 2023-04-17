from dataclasses import dataclass
from datetime import datetime
from typing import List, Protocol

import pandas as pd


@dataclass
class Document:
    """Document is a class that contains a document's title, author, content, and post time."""
    title: str
    author: str
    content: str
    post_time: datetime


class IDocumentsDataset(Protocol):
    """IDocumentsDataset is a protocol that defines the interface of a documents dataset."""
    def __getitem__(self, index: int) -> Document:
        ...

    def __len__(self) -> int:
        ...

    def __next__(self) -> Document:
        ...

    def query_by_time(self, start_time: datetime | str, end_time: datetime | str) -> List[Document]:
        """return a list of document with the time span"""
        ...


class DocumentsDataset(IDocumentsDataset):
    """
    DocumentsDataset is a class that contains a dataset of documents.
    This dataset source can be pandas DataFrame, can supply by csv file or DataFrame, or can supply by document list.
    """

    def __init__(self, documents_csv_path: str | None = None, source_df: pd.DataFrame | None = None,
                 document_list: List[Document] | None = None):
        """supply data source by csv file or DataFrame or document list"""
        # assert only one of the two arguments is not None and both can't be None
        assert (documents_csv_path is not None) or (source_df is not None) or (document_list is not None)
        assert not (documents_csv_path is None and source_df is None and document_list is None)

        if source_df is not None:
            self.documents_df = source_df
        elif documents_csv_path is not None:
            self.documents_df = pd.read_csv(documents_csv_path, index_col=3)
        elif document_list is not None:
            self.documents_df = pd.DataFrame(
                data=[(document.title, document.author, document.content, document.post_time) for document in
                      document_list],
                columns=['title', 'author', 'content', 'post_time'])
            self.documents_df.set_index('post_time', inplace=True)

    def __getitem__(self, index) -> Document:
        row = self.documents_df.iloc[index]

        return Document(title=row['title'], author=row['author'], content=row['content'], post_time=row.name)

    def __len__(self):
        return len(self.documents_df)

    def __iter__(self):
        self.current_index = 0
        return self

    def __next__(self) -> Document:
        if self.current_index >= len(self):
            raise StopIteration
        result = self[self.current_index]
        self.current_index += 1
        return result

    def query_by_time(self, start_time: datetime | str, end_time: datetime | str) -> List[Document]:
        result_df = self.documents_df.loc[start_time:end_time]
        documents = []
        for _, row in result_df.iterrows():
            document = Document(title=row['title'], author=row['author'], content=row['content'], post_time=row.name)
            documents.append(document)
        return documents


if __name__ == "__main__":
    documents_dataset = DocumentsDataset("../organized_data/documents.csv")
    print(documents_dataset.documents_df)
