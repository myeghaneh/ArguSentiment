import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

def flatten_if_nested(y):
    """Flattens a list of lists if needed"""
    if any(isinstance(i, (list, np.ndarray)) for i in y):
        return [item for sublist in y for item in sublist]
    return y

def plot_training_history_cm_cr(history, y_true=None, y_pred=None, classes=None, run_name="run"):
    """
    Plot training/test loss and F1 scores, and print + save confusion matrix and classification report.

    Args:
        history (dict): Dictionary with keys 'train_loss', 'test_loss', 'train_f1', 'test_f1'.
        y_true (list or array, optional): Ground truth labels (can be nested).
        y_pred (list or array, optional): Predicted labels (can be nested).
        classes (list, optional): Class names for classification report.
        run_name (str): Name used for saving output files.
    """
    epochs = range(1, len(history['train_loss']) + 1)

    # === Plot Loss and F1 ===
    plt.figure(figsize=(14, 5))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'bo-', label='Train Loss', linewidth=2)
    plt.plot(epochs, history['test_loss'], 'ro-', label='Test Loss', linewidth=2)
    plt.title('Training and Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xticks(epochs)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()

    # F1
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_f1'], 'bo-', label='Train F1', linewidth=2)
    plt.plot(epochs, history['test_f1'], 'ro-', label='Test F1', linewidth=2)
    plt.title('Training and Test F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.xticks(epochs)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{run_name}_charts.png")
    plt.show()
    print(f"✅ Chart saved as {run_name}_charts.png")

    # === Evaluation ===
    if y_true is not None and y_pred is not None:
        y_true_flat = flatten_if_nested(y_true)
        y_pred_flat = flatten_if_nested(y_pred)

        y_true_flat = np.array(y_true_flat)
        y_pred_flat = np.array(y_pred_flat)

        cm = confusion_matrix(y_true_flat, y_pred_flat)
        cr = classification_report(y_true_flat, y_pred_flat, target_names=classes) if classes else classification_report(y_true_flat, y_pred_flat)

        print("\nConfusion Matrix:")
        print(cm)
        print("\nClassification Report:")
        print(cr)

        with open(f"{run_name}_eval.txt", "w") as f:
            f.write("Confusion Matrix:\n")
            f.write(np.array2string(cm))
            f.write("\n\nClassification Report:\n")
            f.write(cr)

        print(f"✅ Evaluation report saved as {run_name}_eval.txt")
