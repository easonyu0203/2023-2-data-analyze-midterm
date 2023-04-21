from dataclasses import dataclass
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

from datasets.labeled_docs_dataset import ILabeledDataset


@dataclass
class EvaluationResult:
    """
    EvaluationResult is a class that contains the evaluation result of a model.
    """
    model_name: str
    train_accuracy: float
    train_report: str
    train_conf_matrix: str
    val_accuracy: float
    val_report: str
    val_conf_matrix: str


def train_eval_model(model, train_dataset: ILabeledDataset, val_dataset: ILabeledDataset, verbose=True) -> EvaluationResult:
    """
    Train and evaluate a model on the training and validation datasets
    currently, we assume model is regression model and label is continuous
    """
    # Extract features and labels from the training and validation datasets
    X_train, y_train = zip(*train_dataset)
    X_val, y_val = zip(*val_dataset)

    # Train
    if verbose: print("fitting model...")
    model.fit(X_train, y_train)

    if verbose: print("model prediction...")
    # Make predictions on the training and validation datasets
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    # Convert predictions/true_label to binary labels for evaluation
    y_train_pred_binary = _convert_to_binary_labels(y_train_pred)
    y_val_pred_binary = _convert_to_binary_labels(y_val_pred)
    y_train_binary = _convert_to_binary_labels(y_train)
    y_val_binary = _convert_to_binary_labels(y_val)

    # Evaluate the model on the training and validation datasets
    # Calculate performance metrics
    if verbose: print("calculating performance metrics...")
    train_acc = accuracy_score(y_train_binary, y_train_pred_binary)
    val_acc = accuracy_score(y_val_binary, y_val_pred_binary)

    train_report = classification_report(y_train_binary, y_train_pred_binary, target_names=['fallen', 'risen'])
    val_report = classification_report(y_val_binary, y_val_pred_binary, target_names=['fallen', 'risen'])

    train_conf_matrix = confusion_matrix(y_train_binary, y_train_pred_binary)
    val_conf_matrix = confusion_matrix(y_val_binary, y_val_pred_binary)

    result = EvaluationResult(
        model_name=model.__class__.__name__,
        train_accuracy=train_acc,
        train_report=train_report,
        train_conf_matrix=train_conf_matrix,
        val_accuracy=val_acc,
        val_report=val_report,
        val_conf_matrix=val_conf_matrix
    )
    return result


def display_evaluation_result(result: EvaluationResult):
    # Display the results
    print("Training accuracy:", result.train_accuracy)
    print("Validation accuracy:", result.val_accuracy)

    print("\nTraining classification report:")
    print(result.train_report)

    print("\nValidation classification report:")
    print(result.val_report)

    # Plot confusion matrices
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ConfusionMatrixDisplay(result.train_conf_matrix, display_labels=['fallen', 'risen']).plot(ax=ax1, cmap='Blues',
                                                                                              xticks_rotation=45)
    ax1.set_title("Training Confusion Matrix")

    ConfusionMatrixDisplay(result.val_conf_matrix, display_labels=['fallen', 'risen']).plot(ax=ax2, cmap='Blues',
                                                                                            xticks_rotation=45)
    ax2.set_title("Validation Confusion Matrix")

    plt.tight_layout()
    plt.show()


def _convert_to_binary_labels(y_values):
    # Function to convert continuous labels to binary
    return [1 if y > 0 else 0 for y in y_values]
