import matplotlib.pyplot as plt
import seaborn as sns
import os


def plot_target_distribution(target, save_path="plots", show=True):
    os.makedirs(save_path, exist_ok=True)
    plt.figure(figsize=(8, 6))
    sns.countplot(x=target)
    plt.title("Distribution of wine quality scores")
    plt.xlabel("Wine quality")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"{save_path}/target_distribution.png")
    if show:
        plt.show()
    plt.close()


def plot_correlation_matrix(dataframe, save_path="plots", show=True):
    os.makedirs(save_path, exist_ok=True)
    corr = dataframe.corr(numeric_only=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature correlation matrix")
    plt.tight_layout()
    plt.savefig(f"{save_path}/correlation_matrix.png")
    if show:
        plt.show()
    plt.close()


def plot_training_curves(loss, accuracy, model_class, kernel=''):
    fig, ax1 = plt.subplots(figsize=(8,5))

    # loss
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Loss", color="tab:red")
    ax1.plot(loss, color="tab:red", label="Loss")
    ax1.tick_params(axis="y", labelcolor="tab:red")

    # accuracy
    ax2 = ax1.twinx()
    ax2.set_ylabel("Accuracy", color="tab:blue")
    ax2.plot(accuracy, color="tab:blue", label="Accuracy")
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    fig.suptitle(f"Training Curves - {model_class}")
    fig.tight_layout()
    plt.savefig(f"plots/{model_class}_{kernel}.png")
    plt.show()


def plot_confusion_matrix(cm, model):
    class_names = ["Negative", "Positive"]
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig(f"plots/confusion_matrix_{model['model'].__name__}_{(model['grid']['kernel'])}.png")
    plt.show()
