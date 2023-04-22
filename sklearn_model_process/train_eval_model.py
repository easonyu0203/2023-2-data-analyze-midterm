from dataclasses import dataclass
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, roc_auc_score)
from sklearn.metrics import RocCurveDisplay
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
    train_roc_auc: float
    val_accuracy: float
    val_report: str
    val_conf_matrix: str
    val_roc_auc: float


def train_eval_model(model, train_dataset: ILabeledDataset, val_dataset: ILabeledDataset,
                     verbose=True) -> EvaluationResult:
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

    # Evaluate the model on the training and validation datasets
    # Calculate performance metrics
    if verbose: print("calculating performance metrics...")
    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)

    train_roc_auc = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
    val_roc_auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])

    train_report = classification_report(y_train, y_train_pred, target_names=['fallen', 'risen'])
    val_report = classification_report(y_val, y_val_pred, target_names=['fallen', 'risen'])

    train_conf_matrix = confusion_matrix(y_train, y_train_pred, normalize='all')
    val_conf_matrix = confusion_matrix(y_val, y_val_pred, normalize='all')

    result = EvaluationResult(
        model_name=model.__class__.__name__,
        train_accuracy=train_acc,
        train_report=train_report,
        train_conf_matrix=train_conf_matrix,
        train_roc_auc=train_roc_auc,
        val_accuracy=val_acc,
        val_report=val_report,
        val_conf_matrix=val_conf_matrix,
        val_roc_auc=val_roc_auc
    )
    return result


def display_evaluation_result(result: EvaluationResult, model, train_dataset: ILabeledDataset, val_dataset: ILabeledDataset):
    # Display the results
    print("Training accuracy:", result.train_accuracy)
    print("Validation accuracy:", result.val_accuracy)

    print("\nTraining classification report:")
    print(result.train_report)

    print("\nValidation classification report:")
    print(result.val_report)

    # Plot confusion matrices
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

    ConfusionMatrixDisplay(result.train_conf_matrix, display_labels=['fallen', 'risen']).plot(ax=ax1, cmap='Blues',
                                                                                              xticks_rotation=45,
                                                                                              values_format='.2%')
    ax1.set_title("Training Confusion Matrix")

    ConfusionMatrixDisplay(result.val_conf_matrix, display_labels=['fallen', 'risen']).plot(ax=ax2, cmap='Blues',
                                                                                            xticks_rotation=45,
                                                                                            values_format='.2%')
    ax2.set_title("Validation Confusion Matrix")

    plt.tight_layout()
    plt.show()

    # Plot ROC curves and display AUC
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Plot ROC curve for the training dataset
    X_train, y_train = train_dataset.features, train_dataset.labels
    X_val, y_val = val_dataset.features, val_dataset.labels
    RocCurveDisplay.from_estimator(model, X_train, y_train, ax=ax1)
    ax1.set_title(f"Training ROC Curve (AUC = {result.train_roc_auc:.2f})")
    ax1.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)

    # Plot ROC curve for the validation dataset
    RocCurveDisplay.from_estimator(model, X_val, y_val, ax=ax2)
    ax2.set_title(f"Validation ROC Curve (AUC = {result.val_roc_auc:.2f})")
    ax2.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)

    plt.tight_layout()
    plt.show()
