import pandas as pd


class DocumentsDataset:

    def __init__(self, documents_path: str):
        self.documents = pd.read_csv(documents_path, index_col=3)

    def __getitem__(self, index):
        return self.documents.iloc[index]


if __name__ == "__main__":
    documents_dataset = DocumentsDataset("../organized_data/documents.csv")
    print(documents_dataset.documents)
