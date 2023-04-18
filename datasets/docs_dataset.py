from datetime import datetime, timedelta
from typing import List, Protocol, Union, Any

import pandas as pd
from pymongo.collection import Collection
from pymongo.cursor import Cursor

from datasets.document import Document
from db.connect import connect_db


class IDocsDataset(Protocol):
    """IDocumentsDataset is a protocol that defines the interface of documents dataset."""

    def __getitem__(self, index: int) -> Document:
        ...

    def __len__(self) -> int:
        ...

    def __next__(self) -> Document:
        ...

    def query_by_time(self, start_time: datetime | str, end_time: datetime | str) -> Any:
        """return a list of document with the time span (include start_time but exclude end_time)"""
        ...


class DocsDataset(IDocsDataset):
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

        # cache dict for documents
        self.documents_cache = {}

        if source_df is not None:
            self.documents_df = source_df
        elif documents_csv_path is not None:
            self.documents_df = pd.read_csv(documents_csv_path)
            self.documents_df['post_time'] = pd.to_datetime(self.documents_df['post_time'])
            self.documents_df.set_index('post_time', inplace=True)

        elif document_list is not None:
            self.documents_df = pd.DataFrame(
                data=[(document.title, document.author, document.content, document.post_time) for document in
                      document_list],
                columns=['title', 'author', 'content', 'post_time'])
            self.documents_df.set_index('post_time', inplace=True)

    def query_by_time(self, start_time: datetime | pd.Timestamp | str, end_time: datetime | pd.Timestamp | str) \
            -> IDocsDataset:
        result_df = self.documents_df.loc[start_time:end_time - timedelta(seconds=1)]
        documents = []
        for _, row in result_df.iterrows():
            # if the document is already in cache, then return the cached document
            if row.name in self.documents_cache:
                documents.append(self.documents_cache[row.name])
                continue
            else:
                # if the document is not in cache, then create a new document and add it to cache
                document = Document(title=row['title'], author=row['author'], content=row['content'],
                                    post_time=row.name)
                self.documents_cache[row.name] = document
                documents.append(document)
        return DocsDataset(document_list=documents)

    def __getitem__(self, index) -> Document:
        row = self.documents_df.iloc[index]

        # if the document is not in cache, then create a new document and add it to cache
        if row.name not in self.documents_cache:
            self.documents_cache[row.name] = Document(title=row['title'], author=row['author'], content=row['content'],
                                                      post_time=row.name)
        return self.documents_cache[row.name]

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

    def __repr__(self):
        """return the string representation of the DocumentsDataset size of dataset"""
        return f"DocumentsDataset(size={len(self)})"


class DbDocsDataset(IDocsDataset):
    """access data by mongodb"""

    def __init__(self):
        self.collection: Collection = connect_db()
        self.cursor: Cursor = self.collection.find()

    def __getitem__(self, index: int) -> Document:
        doc = self.collection.find().skip(index).limit(1)[0]
        return Document(
            title=doc["title"],
            author=doc["author"],
            content=doc["content"],
            post_time=doc["post_time"],
            keywords=doc.get("keywords", None),
        )

    def __len__(self) -> int:
        return self.collection.count_documents({})

    def __iter__(self):
        return self

    def __next__(self) -> Document:
        if not self.cursor.alive:
            self.cursor = self.collection.find()
        doc = next(self.cursor)
        return Document(
            title=doc["title"],
            author=doc["author"],
            content=doc["content"],
            post_time=doc["post_time"],
            keywords=doc.get("keywords", None),
        )

    def query_by_time(
        self, start_time: Union[datetime, str], end_time: Union[datetime, str]
    ) -> IDocsDataset:
        if isinstance(start_time, str):
            start_time = datetime.fromisoformat(start_time)
        if isinstance(end_time, str):
            end_time = datetime.fromisoformat(end_time)

        docs = self.collection.find({"post_time": {"$gte": start_time, "$lt": end_time}})
        docs = [
                Document(
                    title=doc["title"],
                    author=doc["author"],
                    content=doc["content"],
                    post_time=doc["post_time"],
                    keywords=doc.get("keywords", None),
                )
                for doc in docs
            ]
        return DocsDataset(document_list=docs)


if __name__ == "__main__":
    documents_dataset = DocsDataset("../organized_data/documents.csv")
    print(documents_dataset.documents_df)
