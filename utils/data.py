import random
from tqdm import tqdm

from datasets.docs_dataset import DocsDataset
from datasets.labeled_docs_dataset import ILabeledDataset


def random_split_train_val(dataset: ILabeledDataset | DocsDataset, split_ratio: float = 0.8, verbose=True):
    if verbose:
        print(f"Splitting dataset into training and validation sets with ratio {split_ratio}...")

    if isinstance(dataset, DocsDataset):
        # split df into train and val by split ratio
        train_data = dataset.documents_df.sample(frac=split_ratio)
        val_data = dataset.documents_df.drop(train_data.index)
        train_dataset, val_dataset = DocsDataset(source_df=train_data), DocsDataset(source_df=val_data)
    else:
        # Shuffle the dataset indices
        indices = list(range(len(dataset)))
        random.shuffle(indices)

        # Calculate the number of training samples
        train_size = int(len(dataset) * split_ratio)

        # Split the dataset indices into training and validation sets
        train_indices, val_indices = indices[:train_size], indices[train_size:]

        # Create instances of ILabeledDataset for the training and validation sets
        pbar = tqdm(total=len(train_indices) + len(val_indices), desc="Splitting dataset", disable=not verbose)
        train_data = []
        for idx in train_indices:
            train_data.append(dataset[idx])
            pbar.update(1)
        val_data = []
        for idx in val_indices:
            val_data.append(dataset[idx])
            pbar.update(1)

        # Split the dataset features and labels
        train_features, train_labels = zip(*train_data)
        val_features, val_labels = zip(*val_data)

        # Create instances of the ILabeledDataset subclass with the training and validation data
        train_dataset = dataset.__class__(list(train_features), list(train_labels))
        val_dataset = dataset.__class__(list(val_features), list(val_labels))

    return train_dataset, val_dataset
