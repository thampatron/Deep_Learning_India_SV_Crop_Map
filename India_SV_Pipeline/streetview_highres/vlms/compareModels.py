import os
import csv
import numpy as np
import pandas as pd
import shutil  # for copying images

# IMPORTANT: use a non-interactive backend so figures are saved instead of shown
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

###############################################################################
# CLASS NAME MAP
###############################################################################
CLASS_NAME_MAP = {
    0:  "Wheat or Barley",
    1:  "Peas or Beans or Lentils",
    2:  "Rapeseed",
    3:  "Maize or Sorghum or Millet",
    4:  "Potato",
    5:  "Rice",
    6:  "Soybean or Peanut",
    7:  "Cotton",
    8:  "Sugarcane",
    9:  "Fallow",
    10: "Multiple crop types",
    11: "Too hard to tell",
    12: "Not cropland",
    13: "Other crop",
    14: "Too early in growing stage",
    15: "Image quality poor",
    16: "Grass"
}

# Base folder where images actually reside:
IMAGE_BASE_FOLDER = "/home/laguarta_jordi/sean7391/streetview_highres/highrest_testset_Kharif_2023_3k"

###############################################################################
# 1. Load Ground Truth
###############################################################################
def load_ground_truth(
    gt_csv_path,
    filename_col='image_name',
    classid_col='class_id',
    confidence_col='confidence',
    high_conf_only=False,
    mid_conf_only=False
):
    """
    Loads ground truth from a CSV that has columns:
      - filename_col (default 'image_name')
      - classid_col  (default 'class_id')
      - confidence_col (default 'confidence') used for optional filtering

    If high_conf_only=True, only keep rows with confidence == 'High Confidence'.
    If mid_conf_only=True, only keep rows with confidence in ['High Confidence','Medium Confidence'].

    Returns a dict: gt_map[filename_only] = integer class_id
    """
    gt_map = {}
    with open(gt_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            conf_str = row[confidence_col].strip() if confidence_col in row else ""
            if high_conf_only and conf_str != "High Confidence":
                continue
            if mid_conf_only and conf_str not in ["High Confidence", "Medium Confidence"]:
                continue

            image_path = row[filename_col].strip()
            filename_only = os.path.basename(image_path)

            try:
                class_id = int(row[classid_col])
            except (ValueError, TypeError):
                continue

            # Store only the class ID.
            gt_map[filename_only] = class_id
    
    # Modify class_id 6 to 1
    for filename, class_id in gt_map.items():
        if class_id == 6:
            gt_map[filename] = 1

    return gt_map

###############################################################################
# 2. Load Predictions
###############################################################################
def detect_predicted_label(row):
    """
    Detect which column in the row holds the predicted label:
      - 'Class Number'
      - 'Result Number 1'
    Returns None if neither is found or is empty.
    """
    if 'Class Number' in row and row['Class Number'].strip() != '':
        return int(row['Class Number'].strip())
    elif 'Result Number 1' in row and row['Result Number 1'].strip() != '':
        return int(row['Result Number 1'].strip())
    else:
        return None

def load_model_predictions(csv_paths, filename_col='Original Image Path'):
    """
    Given one or more CSV paths for the same model,
    load them all, detect the predicted label from either
    'Class Number' or 'Result Number 1', and build a dict:
        pred_map[filename_only] = predicted_class_id.
    """
    pred_map = {}
    for path in csv_paths:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Prediction CSV not found: {path}")

        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if filename_col not in row:
                    continue
                image_path = row[filename_col].strip()
                filename_only = os.path.basename(image_path)

                predicted_label = detect_predicted_label(row)
                if predicted_label is not None:
                    pred_map[filename_only] = predicted_label
    return pred_map

###############################################################################
# 3. Alignment & Metrics
###############################################################################
def align_predictions(gt_map, pred_map):
    """
    Returns parallel lists (y_true, y_pred) for filenames found in both
    ground truth (gt_map) and predictions (pred_map).
    """
    y_true, y_pred = [], []
    common_files = set(gt_map.keys()) & set(pred_map.keys())
    for fn in common_files:
        y_true.append(gt_map[fn])
        y_pred.append(pred_map[fn])
    return np.array(y_true), np.array(y_pred)

###############################################################################
# 4. Plotting Helpers (Saving Figures, Not Showing)
###############################################################################
def plot_confusion_matrix(cm, class_ids, model_name, out_dir=".", normalize=False):
    """
    Plot a confusion matrix and save to PNG.
    - cm: 2D numpy array from sklearn.metrics.confusion_matrix
    - class_ids: list of integer class IDs (same order as cm's rows/cols)
    - model_name: name of the model to use in filename/title
    - out_dir: directory to save the figure
    - normalize: if True, display as percentages per row
    """
    class_labels = [CLASS_NAME_MAP.get(cid, str(cid)) for cid in class_ids]

    if normalize:
        cm_sum = cm.sum(axis=1, keepdims=True)
        cm_percent = (cm / (cm_sum + 1e-9)) * 100
        display_matrix = cm_percent
        fmt = ".1f"
        title_ext = " (Normalized %)"
    else:
        display_matrix = cm
        fmt = "d"
        title_ext = " (Counts)"

    fig, ax = plt.subplots()
    im = ax.imshow(display_matrix, cmap='Blues', aspect='auto')
    ax.set_title(f"Confusion Matrix - {model_name}{title_ext}")
    fig.colorbar(im)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(np.arange(len(class_labels)))
    ax.set_yticks(np.arange(len(class_labels)))
    ax.set_xticklabels(class_labels, rotation=45, ha='right')
    ax.set_yticklabels(class_labels)

    # Write text for each cell
    for i in range(len(class_labels)):
        for j in range(len(class_labels)):
            val = display_matrix[i, j]
            ax.text(j, i, f"{val:{fmt}}", ha="center", va="center", color="black")

    plt.tight_layout()

    out_path = os.path.join(out_dir, f"confusion_matrix_{model_name}.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix for {model_name} to {out_path}")

def plot_classwise_f1_bar(classwise_df, out_dir="."):
    """
    Create a grouped bar chart for class-wise F1 scores and save it to PNG.
    classwise_df: DataFrame with index = class IDs, columns = model names, values = F1 scores
    out_dir: where to save the figure
    """
    class_ids = classwise_df.index.tolist()
    model_names = classwise_df.columns.tolist()

    # Convert class_ids to readable names
    class_names = [CLASS_NAME_MAP.get(cid, str(cid)) for cid in class_ids]

    x = np.arange(len(class_ids))
    width = 0.8 / len(model_names)

    fig, ax = plt.subplots()
    for i, model_name in enumerate(model_names):
        f1_values = classwise_df[model_name].values
        ax.bar(x + i * width, f1_values, width, label=model_name)

    ax.set_xlabel("Class")
    ax.set_ylabel("F1 Score")
    ax.set_title("Class-wise F1 Comparison")
    ax.set_xticks(x + width * (len(model_names) / 2 - 0.5))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()

    # Add horizontal dotted lines at increments of 0.1 from 0 to 1.0
    for y in np.arange(0.1, 1.01, 0.1):
        ax.axhline(y=y, color='gray', linestyle='dotted', linewidth=0.7)

    ax.set_ylim([0, 1.05])

    plt.tight_layout()
    out_path = os.path.join(out_dir, "classwise_f1_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved class-wise F1 bar chart to {out_path}")

def plot_classwise_precision_bar(classwise_df, out_dir="."):
    """
    Create a grouped bar chart for class-wise Precision scores and save it to PNG.
    classwise_df: DataFrame with index = class IDs, columns = model names, values = precision scores
    out_dir: where to save the figure
    """
    class_ids = classwise_df.index.tolist()
    model_names = classwise_df.columns.tolist()
    class_names = [CLASS_NAME_MAP.get(cid, str(cid)) for cid in class_ids]
    x = np.arange(len(class_ids))
    width = 0.8 / len(model_names)
    fig, ax = plt.subplots()
    for i, model_name in enumerate(model_names):
        precision_values = classwise_df[model_name].values
        ax.bar(x + i * width, precision_values, width, label=model_name)
    ax.set_xlabel("Class")
    ax.set_ylabel("Precision")
    ax.set_title("Class-wise Precision Comparison")
    ax.set_xticks(x + width * (len(model_names) / 2 - 0.5))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    for y in np.arange(0.1, 1.01, 0.1):
        ax.axhline(y=y, color='gray', linestyle='dotted', linewidth=0.7)
    ax.set_ylim([0, 1.05])
    plt.tight_layout()
    out_path = os.path.join(out_dir, "classwise_precision_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved class-wise Precision bar chart to {out_path}")

def plot_classwise_recall_bar(classwise_df, out_dir="."):
    """
    Create a grouped bar chart for class-wise Recall scores and save it to PNG.
    classwise_df: DataFrame with index = class IDs, columns = model names, values = recall scores
    out_dir: where to save the figure
    """
    class_ids = classwise_df.index.tolist()
    model_names = classwise_df.columns.tolist()
    class_names = [CLASS_NAME_MAP.get(cid, str(cid)) for cid in class_ids]
    x = np.arange(len(class_ids))
    width = 0.8 / len(model_names)
    fig, ax = plt.subplots()
    for i, model_name in enumerate(model_names):
        recall_values = classwise_df[model_name].values
        ax.bar(x + i * width, recall_values, width, label=model_name)
    ax.set_xlabel("Class")
    ax.set_ylabel("Recall")
    ax.set_title("Class-wise Recall Comparison")
    ax.set_xticks(x + width * (len(model_names) / 2 - 0.5))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    for y in np.arange(0.1, 1.01, 0.1):
        ax.axhline(y=y, color='gray', linestyle='dotted', linewidth=0.7)
    ax.set_ylim([0, 1.05])
    plt.tight_layout()
    out_path = os.path.join(out_dir, "classwise_recall_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved class-wise Recall bar chart to {out_path}")

# def plot_overall_metrics_bar(overall_df, out_dir="."):
#     """
#     Create a bar chart where the x-axis = the different metrics (Accuracy, Macro F1,
#     Micro F1, Weighted F1), and each group of bars represents the different models.
#     The legend shows the model names.
#     """
#     # Extract the model names
#     models = overall_df["Model"].tolist()

#     # We have 4 metrics
#     metrics = ["Accuracy", "Macro F1", "Micro F1", "Weighted F1"]

#     data = []
#     for i, model_name in enumerate(models):
#         row = overall_df.loc[i]
#         row_values = [row[m] for m in metrics]
#         data.append(row_values)
#     data = np.array(data)  # shape: (num_models, len(metrics))

#     x = np.arange(len(metrics))  # [0,1,2,3]
#     num_models = len(models)
#     width = 0.8 / num_models

#     fig, ax = plt.subplots()
#     for i, model_name in enumerate(models):
#         offset = i * width
#         ax.bar(x + offset, data[i], width, label=model_name)

#     ax.set_xlabel("Metric")
#     ax.set_ylabel("Score")
#     ax.set_title("Overall Metrics Comparison")
#     ax.set_xticks(x + (num_models - 1) * width / 2)
#     ax.set_xticklabels(metrics)
#     ax.legend()

#     # Add horizontal dotted lines
#     for y in np.arange(0.1, 1.01, 0.1):
#         ax.axhline(y=y, color='gray', linestyle='dotted', linewidth=0.7)

#     ax.set_ylim([0, 1.05])

#     plt.tight_layout()
#     out_path = os.path.join(out_dir, "overall_metrics_comparison.png")
#     plt.savefig(out_path, dpi=150, bbox_inches='tight')
#     plt.close()
#     print(f"Saved overall metrics bar chart to {out_path}")

def plot_stacked_misclassification_by_class(distribution_by_class, all_classes_sorted, n_models, out_dir="."):
    """
    Creates a stacked bar chart where:
      - X-axis: each ground-truth class
      - Y-axis: number of images
      - Each stacked segment represents images misclassified by k=1..n_models
        (i.e., skipping k=0).
      - Legend labels also show the overall percentage (of total images) in brackets.

    distribution_by_class: dict {class_id -> [count_k0, count_k1, ..., count_kn]}
      where count_kX = number of images of class_id misclassified by X models.

    all_classes_sorted: sorted list of class IDs we want on the x-axis.
    n_models: total number of models.
    """
    # We'll skip k=0 (images correct by all models) for this stacked bar plot.
    # But we still need the total count of all images (including k=0) to compute global percentages.

    # 1) Compute total images across all classes and all k
    total_images = 0
    for c in all_classes_sorted:
        total_images += sum(distribution_by_class[c])  # sum of k=0..n_models

    x = np.arange(len(all_classes_sorted))
    width = 0.8

    fig, ax = plt.subplots()
    bottom = np.zeros(len(all_classes_sorted), dtype=int)

    # 2) Loop over misclassification counts k=1..n_models
    for k in range(1, n_models + 1):
        counts_k = []
        global_count_for_k = 0
        for c in all_classes_sorted:
            val = distribution_by_class[c][k]  # how many images of class c are misclassified by k models
            counts_k.append(val)
            global_count_for_k += val

        # Convert to percentage of total images
        if total_images > 0:
            pct = (global_count_for_k / total_images) * 100
        else:
            pct = 0.0

        label_text = f"Misclassified by {k} model(s) ({pct:.1f}%)"

        ax.bar(
            x, counts_k, width,
            label=label_text,
            bottom=bottom
        )
        bottom += np.array(counts_k)

    # 3) Labeling
    class_labels = [CLASS_NAME_MAP.get(cid, str(cid)) for cid in all_classes_sorted]
    ax.set_xticks(x)
    ax.set_xticklabels(class_labels, rotation=45, ha='right')

    ax.set_xlabel("Ground-Truth Class")
    ax.set_ylabel("Number of Images")
    ax.set_title("Stacked Misclassification Count by Class (k=1..n)")
    ax.legend()

    plt.tight_layout()
    out_path = os.path.join(out_dir, "stacked_bar_misclassification_by_class.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved stacked bar chart (excluding k=0) to {out_path}")

###############################################################################
# 5. Master Comparison Function
###############################################################################
def compare_models_with_ground_truth(
    gt_csv_path,
    model_to_csvs,
    out_dir=".",
    filename_col_gt='image_name',
    classid_col_gt='class_id',
    filename_col_pred='Original Image Path',
    confidence_col='confidence',
    high_conf_only=False,
    mid_conf_only=False,
    show_confusion_matrices=True,
    normalize_confusion_matrix=True,
    crops=False
):
    """
    Compare multiple models' predictions against a single ground-truth CSV.

    - gt_csv_path: path to ground truth CSV
    - model_to_csvs: dict {model_name: [list_of_csv_paths_for_that_model]}
    - out_dir: folder to save all figures and outputs
    - filename_col_gt: column in GT CSV containing image path
    - classid_col_gt: column in GT CSV containing the true class ID
    - filename_col_pred: column in prediction CSV containing image path
    - confidence_col: column in GT CSV that might contain 'High Confidence', 'Medium Confidence'
    - high_conf_only: if True, only keep GT labeled as "High Confidence"
    - mid_conf_only: if True, only keep GT labeled as "High Confidence" or "Medium Confidence"
    - show_confusion_matrices: if True, produce confusion matrix figures per model
    - normalize_confusion_matrix: if True, show confusion matrix as row percentages
    - crops: if True, group classes > 9 into class 9 (merging classes 10+ into 9).
    """
    # 1) Load ground truth
    print(f"Loading ground truth from: {gt_csv_path}")
    gt_map = load_ground_truth(
        gt_csv_path=gt_csv_path,
        filename_col=filename_col_gt,
        classid_col=classid_col_gt,
        confidence_col=confidence_col,
        high_conf_only=high_conf_only,
        mid_conf_only=mid_conf_only
    )
    if not gt_map:
        print("No ground truth data found (possibly filtered out?). Exiting.")
        return

    # 2) Load all model predictions
    all_models_predmaps = {}
    for model_name, csv_list in model_to_csvs.items():
        print(f"Loading predictions for model: {model_name}")
        pred_map = load_model_predictions(csv_list, filename_col=filename_col_pred)
        all_models_predmaps[model_name] = pred_map

    # 3) For each model, compute metrics
    overall_metrics = {
        "Model": [],
        "Accuracy": [],
        "Macro F1": [],
        "Micro F1": [],
        "Weighted F1": [],
        "Macro Precision": [],  # Added for precision
        "Micro Precision": [],  # Added for precision
        "Weighted Precision": [],  # Added for precision
        "Macro Recall": [],  # Added for recall
        "Micro Recall": [],  # Added for recall
        "Weighted Recall": []  # Added for recall
    }
    # For class-wise metrics, we now keep separate dictionaries for F1, precision and recall
    classwise_results_f1 = {}       # model_name -> {class_id -> f1}
    classwise_results_precision = {}  # model_name -> {class_id -> precision}
    classwise_results_recall = {}     # model_name -> {class_id -> recall}
    all_classes_set = set()

    for model_name, pred_map in all_models_predmaps.items():
        # Align
        y_true, y_pred = align_predictions(gt_map, pred_map)

        # Possibly group classes > 9 into 9
        if crops:
            y_true = [min(c, 9) for c in y_true]
            y_pred = [min(c, 9) for c in y_pred]

        if len(y_true) == 0:
            print(f"No overlapping filenames with GT for model {model_name}. Skipping metrics.")
            continue

        # 4) Compute overall metrics
        acc = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
        weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Added: Compute precision metrics
        from sklearn.metrics import precision_score, recall_score
        macro_precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        micro_precision = precision_score(y_true, y_pred, average='micro', zero_division=0)
        weighted_precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Added: Compute recall metrics
        macro_recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        micro_recall = recall_score(y_true, y_pred, average='micro', zero_division=0)
        weighted_recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)

        overall_metrics["Model"].append(model_name)
        overall_metrics["Accuracy"].append(acc)
        overall_metrics["Macro F1"].append(macro_f1)
        overall_metrics["Micro F1"].append(micro_f1)
        overall_metrics["Weighted F1"].append(weighted_f1)
        # Added: Store precision metrics
        overall_metrics["Macro Precision"].append(macro_precision)
        overall_metrics["Micro Precision"].append(micro_precision)
        overall_metrics["Weighted Precision"].append(weighted_precision)
        # Added: Store recall metrics
        overall_metrics["Macro Recall"].append(macro_recall)
        overall_metrics["Micro Recall"].append(micro_recall)
        overall_metrics["Weighted Recall"].append(weighted_recall)

        print(f"\nModel {model_name}:")
        print(f"  Accuracy = {acc:.4f}")
        print(f"  Macro-F1 = {macro_f1:.4f}")
        print(f"  Micro-F1 = {micro_f1:.4f}")
        print(f"  Weighted-F1 = {weighted_f1:.4f}")
        # Added: Print precision metrics
        print(f"  Macro-Precision = {macro_precision:.4f}")
        print(f"  Micro-Precision = {micro_precision:.4f}")
        print(f"  Weighted-Precision = {weighted_precision:.4f}")
        # Added: Print recall metrics
        print(f"  Macro-Recall = {macro_recall:.4f}")
        print(f"  Micro-Recall = {micro_recall:.4f}")
        print(f"  Weighted-Recall = {weighted_recall:.4f}")

        # 5) Class-wise Metrics (F1, Precision, Recall)
        cr = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        unique_c = np.unique(y_true)
        model_classwise_f1 = {}
        model_classwise_precision = {}
        model_classwise_recall = {}
        for c in unique_c:
            c_str = str(c)
            if c_str in cr:
                model_classwise_f1[c] = cr[c_str]["f1-score"]
                model_classwise_precision[c] = cr[c_str]["precision"]
                model_classwise_recall[c] = cr[c_str]["recall"]
            else:
                model_classwise_f1[c] = 0.0
                model_classwise_precision[c] = 0.0
                model_classwise_recall[c] = 0.0
        classwise_results_f1[model_name] = model_classwise_f1
        classwise_results_precision[model_name] = model_classwise_precision
        classwise_results_recall[model_name] = model_classwise_recall
        all_classes_set.update(unique_c)

        # 6) Confusion Matrix
        if show_confusion_matrices:
            sorted_cids = sorted(all_classes_set)
            cm = confusion_matrix(y_true, y_pred, labels=sorted_cids)
            plot_confusion_matrix(
                cm,
                class_ids=sorted_cids,
                model_name=model_name,
                out_dir=out_dir,
                normalize=normalize_confusion_matrix
            )

    # 7) Summarize Overall
    if len(overall_metrics["Model"]) == 0:
        print("No valid models to summarize.")
        return

    overall_df = pd.DataFrame(overall_metrics)
    print("\n=== Overall Metrics by Model ===")
    print(overall_df.round(4))

    # 8) Class-wise Metrics DataFrames
    all_classes_sorted = sorted(all_classes_set)
    rows_f1 = []
    rows_precision = []
    rows_recall = []
    for c in all_classes_sorted:
        row_f1 = {"Class": c}
        row_precision = {"Class": c}
        row_recall = {"Class": c}
        for model_name in classwise_results_f1.keys():
            row_f1[model_name] = classwise_results_f1[model_name].get(c, 0.0)
            row_precision[model_name] = classwise_results_precision[model_name].get(c, 0.0)
            row_recall[model_name] = classwise_results_recall[model_name].get(c, 0.0)
        rows_f1.append(row_f1)
        rows_precision.append(row_precision)
        rows_recall.append(row_recall)

    classwise_df_f1 = pd.DataFrame(rows_f1).set_index("Class")
    classwise_df_precision = pd.DataFrame(rows_precision).set_index("Class")
    classwise_df_recall = pd.DataFrame(rows_recall).set_index("Class")
    print("\n=== Class-wise F1 Scores ===")
    print(classwise_df_f1.round(4))

    # 9) Plot classwise & overall comparisons
    plot_classwise_f1_bar(classwise_df_f1, out_dir=out_dir)
    plot_classwise_precision_bar(classwise_df_precision, out_dir=out_dir)
    plot_classwise_recall_bar(classwise_df_recall, out_dir=out_dir)
    plot_overall_metrics_bar(overall_df, out_dir=out_dir)

    ###########################################################################
    # 10) NEW FEATURE: Calculate how many models got each image wrong vs right
    #     Then show distribution & save "all-wrong" images grouped by GT class.
    ###########################################################################

    model_names = list(model_to_csvs.keys())
    n_models = len(model_names)

    # 10a) Find the set of files that exist in *all* models' predictions
    common_files_all = set(gt_map.keys())
    for m in model_names:
        common_files_all &= set(all_models_predmaps[m].keys())

    if not common_files_all:
        print("\nNo images found in *all* model predictions. Skipping distribution analysis.")
        return

    # 10b) For each file in that intersection, check how many models predicted it correctly
    correct_count_map = {}
    for fn in common_files_all:
        gt_class = gt_map[fn]
        if crops:
            gt_class = min(gt_class, 9)

        correct_count = 0
        for m in model_names:
            pred_class = all_models_predmaps[m][fn]
            if crops:
                pred_class = min(pred_class, 9)
            if pred_class == gt_class:
                correct_count += 1
        correct_count_map[fn] = correct_count

    # 10c) Build distribution: how many images are misclassified by k models?
    misclassified_distribution = {k: 0 for k in range(n_models + 1)}
    # track which are "all-wrong" (k == n_models)
    all_wrong_files = []

    # For the stacked bar by class, track distribution_by_class:
    distribution_by_class = {}
    for c in all_classes_sorted:
        distribution_by_class[c] = [0]*(n_models+1)  # from 0..n_models

    for fn, ccount in correct_count_map.items():
        # number of models that got it wrong:
        wrong_count = n_models - ccount
        misclassified_distribution[wrong_count] += 1
        if wrong_count >= n_models-1:
            all_wrong_files.append(fn)

        # Also update distribution_by_class
        # Must re-derive the GT class from the dictionary
        gt_class = gt_map[fn]
        if crops:
            gt_class = min(gt_class, 9)
        distribution_by_class[gt_class][wrong_count] += 1

    # 10d) Print distribution stats
    print("\n=== Distribution of how many models got an image WRONG ===")
    total = len(common_files_all)
    print(f"Total images considered: {total}")
    for k in range(n_models + 1):
        count_k = misclassified_distribution[k]
        pct = (count_k / total) * 100.0
        print(f"Wrong by {k:2d} model(s): {count_k:4d} image(s)  => {pct:6.2f}%")

    # 10d.5) Plot stacked bar chart by class
    plot_stacked_misclassification_by_class(distribution_by_class, all_classes_sorted, n_models, out_dir)

    # 10e) Save "all-wrong" images into subfolders by ground-truth class
    if all_wrong_files:
        images_root = os.path.join(out_dir, "images")
        os.makedirs(images_root, exist_ok=True)

        for fn in all_wrong_files:
            # Determine the GT class
            gt_class = gt_map[fn]
            if crops:
                gt_class = min(gt_class, 9)

            class_str = CLASS_NAME_MAP.get(gt_class, str(gt_class))
            class_folder = os.path.join(images_root, class_str)
            os.makedirs(class_folder, exist_ok=True)

            # Use the known base folder path + filename
            src_path = os.path.join(IMAGE_BASE_FOLDER, fn)
            dst_path = os.path.join(class_folder, fn)

            if os.path.isfile(src_path):
                try:
                    shutil.copy2(src_path, dst_path)
                except Exception as e:
                    print(f"Could not copy {src_path} to {dst_path}: {e}")
            else:
                print(f"File not found: {src_path}; skipping copy")

        print(f"\nSaved {len(all_wrong_files)} all-wrong images into '{images_root}/<className>/' subfolders.")
    else:
        print("\nNo images were misclassified by ALL models.")

# Now update the plot_overall_metrics_bar function to include the new metrics
def plot_overall_metrics_bar(overall_df, out_dir="."):
    """
    Create a bar chart where the x-axis = the different metrics (Accuracy, Macro F1,
    Micro F1, Weighted F1, Macro Precision, Micro Precision, Weighted Precision,
    Macro Recall, Micro Recall, Weighted Recall), and each group of bars represents 
    the different models.
    The legend shows the model names.
    """
    # Extract the model names
    models = overall_df["Model"].tolist()

    # Updated: We now have 10 metrics (original 4 plus 6 new ones)
    metrics = [
        "Accuracy", 
        "Macro F1", "Micro F1", "Weighted F1",
        "Macro Precision", "Micro Precision", "Weighted Precision",
        "Macro Recall", "Micro Recall", "Weighted Recall"
    ]

    data = []
    for i, model_name in enumerate(models):
        row = overall_df.loc[i]
        row_values = [row[m] for m in metrics]
        data.append(row_values)
    data = np.array(data)  # shape: (num_models, len(metrics))

    x = np.arange(len(metrics))
    num_models = len(models)
    width = 0.8 / num_models

    fig, ax = plt.subplots(figsize=(15, 8))  # Larger figure size to accommodate more metrics
    for i, model_name in enumerate(models):
        offset = i * width
        ax.bar(x + offset, data[i], width, label=model_name)

    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")
    ax.set_title("Overall Metrics Comparison")
    ax.set_xticks(x + (num_models - 1) * width / 2)
    ax.set_xticklabels(metrics, rotation=45, ha='right')  # Rotate labels for better readability
    ax.legend()

    # Add horizontal dotted lines
    for y in np.arange(0.1, 1.01, 0.1):
        ax.axhline(y=y, color='gray', linestyle='dotted', linewidth=0.7)

    ax.set_ylim([0, 1.05])

    plt.tight_layout()
    out_path = os.path.join(out_dir, "overall_metrics_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved overall metrics bar chart to {out_path}")

###############################################################################
# 6. Example Usage (if run as a script)
###############################################################################
if __name__ == "__main__":
    # Example usage:
    # 1) Path to your ground-truth CSV:
    #    Must have columns like: ["image_name", "class_id", "confidence" (optional)].
    ground_truth_csv = "/home/laguarta_jordi/sean7391/streetview_highres/vlms/labels1/cropLabelsBatchAllProcessedFlooded.csv"

    # Put them in a dict mapping model_name -> list of CSV paths:
    models = [ 'gpt4-benPrompt', 'gemini-2.5-flash-preview-04-17-crop75', 'gemini-2.0-flash-benPrompt-doubt','gemini-2.0-flash-StructuredPrompt', 'gemini-2.0-flash-StructuredPromptLegumesAmiguous', 'gemini-2.0-flash-StructuredPromptLegumesAmiguous_crop75', 'gemini-2.0-flash-StructuredPromptLegumesAmiguous_crop50']
    model_csvs = {}
    for model in models:
            path1 = f'/home/laguarta_jordi/sean7391/streetview_highres/vlms/labelsBatch1/{model}/kharif_2023_1k_results.csv'
            path2 = f'/home/laguarta_jordi/sean7391/streetview_highres/vlms/labelsBatch2/{model}/kharif_2023_3k_results.csv'
            model_csvs[model] = [path1, path2]

    # 3) Output directory for saving figures
    output_folder = '/home/laguarta_jordi/sean7391/streetview_highres/vlms/labelsBatchAll/comparedModels'

    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    # 4) Compare them
    compare_models_with_ground_truth(
        gt_csv_path=ground_truth_csv,
        model_to_csvs=model_csvs,
        out_dir=output_folder,
        filename_col_gt='image_name',       # Column in GT CSV for file path
        classid_col_gt='class_id',          # Column in GT CSV for true class
        filename_col_pred='Original Image Path',  # Column in predictions CSV for file path
        confidence_col='confidence',        # If your GT CSV has this column
        high_conf_only=False,
        mid_conf_only=False,
        show_confusion_matrices=True,       # Save confusion matrices per model
        normalize_confusion_matrix=True,    # Plot CM as percentages
        crops=True                          # Merge classes >9 into class 9
    )
