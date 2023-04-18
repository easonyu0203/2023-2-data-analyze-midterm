import random

from datasets.labeled_docs_dataset import ILabeledDataset


def random_split_train_val(dataset: ILabeledDataset, split_ratio: float = 0.8):
    # Shuffle the dataset indices
    indices = list(range(len(dataset)))
    random.shuffle(indices)

    # Calculate the number of training samples
    train_size = int(len(dataset) * split_ratio)

    # Split the dataset indices into training and validation sets
    train_indices, val_indices = indices[:train_size], indices[train_size:]

    # Create instances of ILabeledDataset for the training and validation sets
    train_data = [dataset[idx] for idx in train_indices]
    val_data = [dataset[idx] for idx in val_indices]

    # Split the dataset features and labels
    train_features, train_labels = zip(*train_data)
    val_features, val_labels = zip(*val_data)

    # Create instances of the ILabeledDataset subclass with the training and validation data
    train_dataset = dataset.__class__(list(train_features), list(train_labels))
    val_dataset = dataset.__class__(list(val_features), list(val_labels))

    return train_dataset, val_dataset
