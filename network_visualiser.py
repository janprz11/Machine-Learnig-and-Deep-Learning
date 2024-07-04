import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def make_plots_history(model, type, x_test, y_test, history, variant):
    plt.figure()
    plt.plot(history.history[f"accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.legend(["Training Accuracy", "Validation Accuracy"])
    plt.xlabel("Training epoch")
    plt.ylabel("Accuracy value")
    plt.grid(True, "both")
    plt.tight_layout()
    plt.savefig(f"img\\NETS_{type}\\{variant}_ACCURACY.png")

    plt.figure()
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.legend(["Training Loss", "Validation Loss"])
    plt.xlabel("Training epoch")
    plt.ylabel("Loss value")
    plt.grid(True, "both")
    plt.tight_layout()
    plt.savefig(f"img\\NETS_{type}\\{variant}_LOSS.png")

    preds = model.predict(x_test)
    y_pred_binary = np.round(preds)
    cm = confusion_matrix(y_test, y_pred_binary)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Short anchor", "Long anchor"])
    disp.plot()
    plt.tight_layout()
    plt.savefig(f"img\\NETS_{type}\\{variant}_CONF_MAT.png")

    return [balanced_accuracy_score(y_test, y_pred_binary)]

def export_data_to_excel(losses, type, accuracies, variant):
    df=pd.DataFrame() 
    df["losses"] = losses
    df["accuracies"] = accuracies
    df.to_excel(f"results_{type}\KFOLD_RESULTS_{variant}.xlsx", index=False)