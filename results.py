from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
import pandas as pd
import argparse
import wandb
from app.const import WANDB_PROJECT, TARGET_COLUMN, PREDICTED_COLUMN, DEFAULT_CSV_PREDICTIONS_TRAIN_PATH
import logging

#logging.basicConfig(
#    level=logging.INFO,
#    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
#)

def plot_pr_curve(y_true, y_pred):
    print("Plotting PR curve...")
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
