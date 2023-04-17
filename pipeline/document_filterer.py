from typing import Protocol
from datasets.documents_dataset import IDocumentsDataset, DocumentsDataset
from datasets.stock_dataset import Stock


class IDocumentsFilterer(Protocol):
    """Filter documents that aren't relevant to the stock we are interested in."""
    def filter_documents(self, documents: IDocumentsDataset, stock: Stock) -> IDocumentsDataset:
        """filter documents by relevant to the stock we are interested in."""
        ...


class DefaultFilterer(IDocumentsFilterer):
    """use whether doc title or content contains the stock name to filter documents"""
    def filter_documents(self, documents: IDocumentsDataset, stock: Stock) -> IDocumentsDataset:
        stock_name = stock.name
        filtered_documents = []
        for document in documents:
            if stock_name in document.title or stock_name in document.content:
                filtered_documents.append(document)

        return DocumentsDataset(document_list=filtered_documents)

