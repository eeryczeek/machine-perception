import os
import json
import pandas as pd
from plotnine import ggplot, aes, geom_line, labs, theme_minimal, theme, element_blank, scale_y_continuous, scale_x_continuous


def load_history(json_path="training_history.json"):
    with open(json_path, "r") as f:
        history = json.load(f)
    return pd.DataFrame({
        "epoch": list(range(1, len(history["train_loss"]) + 1)),
        "train_loss": history["train_loss"],
        "val_loss": history["val_loss"],
        "val_accuracy": history["val_accuracy"]
    })


def get_custom_theme():
    return (
        theme_minimal(base_size=14)
        + theme(
            panel_background=element_blank(),
            plot_background=element_blank(),
            figure_size=(8, 5)
        )
    )


def plot_train_loss(df, out_dir):
    p = (
        ggplot(df, aes(x="epoch", y="train_loss"))
        + geom_line(color="red", size=1)
        + labs(title="Train Loss", x="Epoch", y="Loss")
        + scale_x_continuous(breaks=range(1, df["epoch"].max() + 1))
        + get_custom_theme()
    )
    p.save(os.path.join(out_dir, "train_loss.png"))
    print("Train Loss plot saved.")


def plot_val_loss(df, out_dir):
    p = (
        ggplot(df, aes(x="epoch", y="val_loss"))
        + geom_line(color="green", size=1)
        + labs(title="Validation Loss", x="Epoch", y="Loss")
        + scale_x_continuous(breaks=range(1, df["epoch"].max() + 1))
        + get_custom_theme()
    )
    p.save(os.path.join(out_dir, "val_loss.png"))
    print("Validation Loss plot saved.")


def plot_val_accuracy(df, out_dir):
    p = (
        ggplot(df, aes(x="epoch", y="val_accuracy"))
        + geom_line(color="blue", size=1)
        + labs(title="Validation Accuracy", x="Epoch", y="Accuracy (%)")
        + scale_y_continuous(limits=(0, 100))
        + scale_x_continuous(breaks=range(1, df["epoch"].max() + 1))
        + get_custom_theme()
    )
    p.save(os.path.join(out_dir, "val_accuracy.png"))
    print("Validation Accuracy plot saved.")


def main():
    out_dir = "plots"
    os.makedirs(out_dir, exist_ok=True)
    df = load_history()
    plot_train_loss(df, out_dir)
    plot_val_loss(df, out_dir)
    plot_val_accuracy(df, out_dir)


if __name__ == "__main__":
    main()