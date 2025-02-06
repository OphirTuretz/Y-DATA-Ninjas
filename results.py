from sklearn.metrics import (classification_report, accuracy_score, f1_score, precision_score, recall_score,
                             precision_recall_curve, auc)
import pandas as pd
import numpy as np
import argparse
import wandb
from app.const import WANDB_PROJECT, TARGET_COLUMN, PREDICTED_COLUMN, DEFAULT_CSV_PREDICTIONS_TRAIN_PATH, REVENUE_COST_DICT
import logging
import matplotlib.pyplot as plt


#logging.basicConfig(
#    level=logging.INFO,
#    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
#)
def log_profit(y_true, y_pred, precisions, recalls, thresholds):
    # Calculate profit from revenue and cost
    print("Calculating profit...")
    revenue = REVENUE_COST_DICT['revenue']
    cost = REVENUE_COST_DICT['cost']
    tp = recalls * sum(y_true)  # Recall = TP / (TP + FN) → TP = Recall * (TP + FN)
    fp = (tp / (precisions + 1e-10)) - tp
    profits = tp * revenue - fp * cost

    # Find the threshold that maximizes profit
    max_profit_idx = np.argmax(profits)
    max_profit = profits[max_profit_idx]
    max_profit_threshold = thresholds[max_profit_idx]

    run.log({
        "max_profit": max_profit,
        "max_profit_threshold": max_profit_threshold
    })
    print("Profit calculated and logged.")

    # Log a short summary of the profit
    summary = f"""
    ## Profit Optimization Summary

    In our current scenario:  
    - **Revenue per correct prediction:** $ {revenue}  
    - **Cost per false positive:** $ {cost}  

    The **maximum profit** of **${max_profit:.2f}** is achieved at a **threshold of {max_profit_threshold:.2f}**.  

    With this optimal threshold:  
    - ✅ **True Positives:** {tp[max_profit_idx]:.0f}  
    - ❌ **False Positives:** {fp[max_profit_idx]:.0f} """

    print(summary)
    run.log({"profit_summary": wandb.Html(summary)}) # Can be found in the file/media tab of the run

    return profits, max_profit_threshold

def plot_profit_per_threshold(precisions, recalls, thresholds, profits, max_profit_threshold):

    # Calculate F1 Score for plotting
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    optimal_threshold_for_f1 = thresholds[np.argmax(f1_scores)]

    # Plot Profit vs. Threshold
    plt.plot(thresholds, profits[:-1], label="Profit Curve", color='blue')
    plt.axvline(x=max_profit_threshold, color='red', linestyle="--", 
                label=f"Best Threshold = {max_profit_threshold:.2f}")
    
    plt.xlabel("Probability Threshold")
    plt.ylabel("Profit")
    plt.title("Optimal Threshold for Maximum Profit")
    plt.legend(fontsize='small', loc='best')

    # Plot F1 Score vs. Threshold
    plt.twinx()
    plt.plot(thresholds, f1_scores[1:], label="F1 Score", color='green', alpha=0.5)
    plt.axvline(x=optimal_threshold_for_f1, color='orange', linestyle="--", 
                label=f"Optimal Threshold for F1 = {optimal_threshold_for_f1:.2f}", alpha=0.5 )
    plt.ylabel("F1 Score")
    
    plt.legend(fontsize='small', loc='lower left')
    run.log({"profit_vs_threshold": wandb.Image(plt)})
    plt.close()

    return optimal_threshold_for_f1


def plot_pr_curve(y_true, y_pred):
    print("Plotting PR curve...")
    y_pred = np.vstack([1 - y_pred, y_pred]).T # Add probabilities for the negative class
    wandb.log({"pr": wandb.plot.pr_curve(y_true, y_pred)})
    print("PR curve plotted.")


def plot_confusion_matrix(y_true, y_pred, labels=[0, 1]):
    print("Plotting confusion matrix...")
    wandb.sklearn.plot_confusion_matrix(y_true, y_pred, labels)
    print("Confusion matrix plotted.")


def plot_classification_report(y_true, y_pred):
    print("Plotting classification report...")

    # Return a classification report results dict
    report = classification_report(y_true, y_pred, output_dict=True)
    # Convert the report to a DataFrame
    report_df = pd.DataFrame(report).transpose()
    # Convert the DataFrame to a W&B table
    classification_table = wandb.Table(dataframe=report_df)
    # Log the table to W&B
    wandb.log({"classification_report": classification_table})

    print("Classification report plotted.")


def extract_correct_and_predicted_targets(df):
    print("Extracting correct and predicted targets...")
    y_true = df[TARGET_COLUMN]
    y_pred = df[PREDICTED_COLUMN]
    print("Correct and predicted targets extracted.")
    return y_true, y_pred


def load_data(path: str) -> pd.DataFrame:
    print(f"Loading data from {path}...")
    df = pd.read_csv(path)
    print(df.info())
    print("Data loaded.")
    return df


if __name__ == "__main__":
    print("results.py started...")

    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input-data-path", default=DEFAULT_CSV_PREDICTIONS_TRAIN_PATH)
    parser.add_argument("-wgid", "--wandb-group-id", default=None)

    args = parser.parse_args()

    # Resume the run
    api = wandb.Api()
    runs = api.runs(f"Y-DATA-Ninjas/{WANDB_PROJECT}")
    runs_list = list(runs)  # Convert to a list
    last_run = runs_list[-1] 
    run = wandb.init(entity="Y-DATA-Ninjas", project=WANDB_PROJECT, id=last_run.id, resume="must")
    
    df = load_data(args.input_data_path)
    y_true, y_pred = extract_correct_and_predicted_targets(df)

    #plot_classification_report(y_true, y_pred)

    #plot_confusion_matrix(y_true, y_pred)

    # plot_pr_curve(y_true, y_pred)

    # TODO: upload prediction errors as tables to W&B
    # TODO: upload charts:
    # im = PIL.fromarray()
    # rgb_im = im.convert('RGB')
    # rgb_im.save('my_image.jpg')
    # wandb.log({"example_image": wandb.Image('my_image.jpg')})

    #logging.info("Evaluating model on test set...")
    #report = classification_report(y_true, y_pred, output_dict=True)

    if set(y_pred.unique()) == {0, 1} or set(y_pred.unique()) == {0} or set(y_pred.unique()) == {1}:
        print("Calculating metrics for binary classification...")
        f1=f1_score(y_true, y_pred)
        precision=precision_score(y_true, y_pred)
        recall=recall_score(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)

        run.log({
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "accuracy": accuracy
        })

    else:
        print("Calculating metrics for probabilities - PR curve...")

        # Calculate PR AUC
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred)
        auc_score = auc(recalls, precisions)
        run.log({"pr_auc": auc_score})

        # Plot PR curve
        plot_pr_curve(y_true, y_pred)

        # Calculate profit and log it
        profits, max_profit_threshold = log_profit(y_true, y_pred, precisions, recalls, thresholds)

        # Plot profit vs. threshold
        optimal_thrshold_for_f1 =  plot_profit_per_threshold(precisions, recalls, thresholds, profits, max_profit_threshold)


        print("Binarizing predictions and calculating metrics for binary classification...")
        y_pred = np.where(y_pred > optimal_thrshold_for_f1, 1, 0)
        f1=f1_score(y_true, y_pred)
        precision=precision_score(y_true, y_pred)
        recall=recall_score(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)

        run.log({
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "accuracy": accuracy
        })


    #logging.info(f"Accuracy: {accuracy:.4f}")
    #logging.info(f"f1: {f1:.4f}")
    #logging.info(f"precision: {precision:.4f}")
    #logging.info(f"recall: {recall:.4f}")
    # logging.info("Classification Report:")
    # logging.info(classification_report(y_true, y_pred))

    
    run.finish()
    print("results.py finished.")
