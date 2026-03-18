import csv
import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    classification_report
)

# ---------------------------------------------------------------------------
# 1. Class Name Mapping
# ---------------------------------------------------------------------------

# Define your class ID -> class name mapping (0 to 14)
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
    15: "Image quality poor"
}



# ---------------------------------------------------------------------------
# 2. Parsing CSVs
# ---------------------------------------------------------------------------


def analyze_predictions(csv_path):
    all_same_count = 0       # For rows where all 3 are the same
    two_same_count = 0       # For rows where exactly 2 match
    all_different_count = 0  # For rows where all 3 are different
    total_rows = 0

    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Extract the 3 result columns
            r1 = row['Result Number 1'].strip()
            r2 = row['Result Number 2'].strip()
            r3 = row['Result Number 3'].strip()
            
            # skip row if any are empty (optional check)
            # if not r1 or not r2 or not r3:
            #     continue

            total_rows += 1
            
            results = [r1, r2, r3]
            unique_values = set(results)

            if len(unique_values) == 1:
                # All the same
                all_same_count += 1
            elif len(unique_values) == 2:
                # Exactly 2 are the same, because out of 3 total, 
                # if there are 2 unique values, one of them appears twice
                two_same_count += 1
            else:
                # len(unique_values) == 3 => all different
                all_different_count += 1
    
    # Now print percentages:
    if total_rows > 0:
        print("Total rows:", total_rows)
        print("All 3 same:         {:.2f}%".format(100.0 * all_same_count / total_rows))
        print("Exactly 2 the same: {:.2f}%".format(100.0 * two_same_count / total_rows))
        print("All 3 different:    {:.2f}%".format(100.0 * all_different_count / total_rows))
    else:
        print("No rows found in CSV.")

def load_ground_truth(gt_csv_path, high_conf=False, mid_conf=False):
    """
    Ground truth CSV columns (example):
      confidence,crop_type,image_name,class_id

    Returns a dict:
      filename_only -> integer class_id
    """
    gt_map = {}
    with open(gt_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        i = 0
        for row in reader:
            i+=1
            if i > 100:
                # If we want to filter out anything that is not "High Confidence"
                if high_conf and row['confidence'].strip() != "High Confidence":
                    continue
                if mid_conf and row['confidence'].strip() not in ["High Confidence", "Medium Confidence"]:
                    continue

                image_name = row['image_name'].strip()
                filename_only = os.path.basename(image_name)
                class_id = int(row['class_id'])
                gt_map[filename_only] = class_id
    return gt_map

def majority_vote(*values):
    """
    Returns the 'majority vote' among any odd number of values.
    
    If there is a single clear winner (strictly higher count than all others),
    return that. If there is a tie, return the first one in the original list.

    Examples:
      majority_vote(5,5,5)   -> 5
      majority_vote(5,5,3)   -> 5
      majority_vote(3,5,7)   -> 3  (all different => tie => pick first)
      majority_vote(3,5,5,7,7) -> 5 or 7 could be tied for top.
                               => picks whichever appears first, in this example: 5
    """
    if not values:
        return None  # or raise an Exception if no values provided

    counter = Counter(values)
    max_count = max(counter.values())  # highest frequency
    # Gather all values with max_count
    candidates = [val for val, cnt in counter.items() if cnt == max_count]

    if len(candidates) == 1:
        # Exactly one unique winner
        return candidates[0]
    else:
        # Tie => return whichever appears first in original *values
        for v in values:
            if v in candidates:
                return v

def voting_from_row(row, vote_cols):
    """
    Given a CSV 'row' (a dict from csv.DictReader) and a list of column names,
    extract each column’s integer vote, then call 'majority_vote' to get
    the final label.

    :param row:       One dictionary row from a CSV, e.g. row['Result Number 1'] = '5'
    :param vote_cols: A list of string column names to use as votes,
                      e.g. ["Result Number 1", "Result Number 2", "Result Number 3"]
    :return:          The final integer label chosen by the majority
                      (or the first if there’s a tie).
                      Returns None if no valid columns / no votes found.
    """
    votes = []
    for col in vote_cols:
        val_str = row[col].strip() if col in row else ''
        if val_str:
            votes.append(int(val_str))

    if not votes:
        return None  # or raise an Exception if no valid data

    # Pass all votes to 'majority_vote'
    final_label = majority_vote(*votes)
    return final_label                

def load_predictions(pred_csv_path, label_column_names):
    """
    Prediction CSV columns (example):
      Original Image Path,Result Number

    Returns a dict:
      filename_only -> integer class_id
    """
    pred_map = {}
    with open(pred_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            original_path = row['Original Image Path'].strip()
            filename_only = os.path.basename(original_path)

            if len(label_column_names) < 2:
                result_number = row[label_column_names[0]]
                
                # Skip rows with empty Result Number
                if result_number.strip() == '':
                    # print(f"Skipping row with empty Result Number: {image_path}")
                    continue

                predicted_class_id = int(row[label_column_names[0]])

            else:
                predicted_class_id = voting_from_row(row, label_column_names)

            pred_map[filename_only] = predicted_class_id
    return pred_map


# ---------------------------------------------------------------------------
# 3. Evaluation
# ---------------------------------------------------------------------------

def evaluate_predictions(gt_map, pred_map):
    """
    Build aligned lists of ground-truth and predicted class IDs for
    all filenames appearing in BOTH dictionaries.

    Returns (y_true, y_pred, matched_filenames)
    """
    y_true, y_pred, matched_files = [], [], []

    # Intersection of filenames
    common_filenames = set(gt_map.keys()) & set(pred_map.keys())

    for fn in common_filenames:
        y_true.append(gt_map[fn])
        y_pred.append(pred_map[fn])
        matched_files.append(fn)

    return y_true, y_pred, matched_files

def plot_confusion_matrix(cm, class_names,
                          savepath='confusion_matrix.png',
                          title='Confusion Matrix',
                          percent=False):
    """
    Plots and saves a confusion matrix figure using matplotlib.
    
    Parameters:
      cm          (np.array): 2D confusion matrix (absolute counts)
      class_names (list):     list of class names (strings) for axes
      savepath    (str):      output file path
      title       (str):      title for the plot
      percent     (bool):     if True, plot row-based percentages rather than absolute counts
    """
    import itertools
    import matplotlib.pyplot as plt
    import numpy as np

    # If percent == True, convert matrix to row-wise percentages
    if percent:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_to_plot = (cm / (row_sums + 1e-9)) * 100.0
        fmt = '.1f'  # e.g. "12.3" for 12.3%
    else:
        cm_to_plot = cm
        fmt = 'd'   # integer format

    ##############################################################################
    # ADDED: Compute totals for each row & column (in absolute counts)
    #        and build an (N+1)x(N+1) "expanded" matrix.
    ##############################################################################
    n = cm.shape[0]  # number of classes
    row_totals = cm.sum(axis=1)   # absolute total for each true class (row)
    col_totals = cm.sum(axis=0)   # absolute total for each predicted class (col)
    grand_total = cm.sum()

    # Make a new (N+1)x(N+1) array for plotting:
    # Top-left NxN block => your original row-wise % confusion matrix
    # Last row/col       => store zeros here so they don't affect the color scale;
    #                       we'll overlay text with actual totals.
    expanded_cm = np.zeros((n+1, n+1), dtype=float)
    expanded_cm[:n, :n] = cm_to_plot

    # Also extend the class_names list by one ("Total"):
    class_names_extended = list(class_names) + ["Total"]
    ##############################################################################

    # Increase figure size a bit to accommodate the extra row/column
    plt.figure(figsize=(9, 7))
    
    # Plot the expanded matrix
    plt.imshow(expanded_cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()

    # Right after you do "plt.imshow(...)" and "plt.colorbar()", add:
    plt.axvline(x=n - 0.5, color='black', linewidth=2)
    plt.axhline(y=n - 0.5, color='black', linewidth=2)

    ##############################################################################
    # Update tick marks to go from 0..n (which is now n+1 positions)
    ##############################################################################
    tick_marks = np.arange(n+1)
    plt.xticks(tick_marks, class_names_extended, rotation=45, ha='right')
    plt.yticks(tick_marks, class_names_extended)

    # For threshold-based text color, we only want the max from the main block
    main_block_max = cm_to_plot.max() if (cm_to_plot.size > 0) else 0
    thresh = main_block_max / 2.0
    ##############################################################################

    # Write the values in each cell
    for i, j in itertools.product(range(n+1), range(n+1)):
        # Case 1: Inside the main NxN confusion region
        if i < n and j < n:
            value = expanded_cm[i, j]
            # Decide text color based on threshold
            color = "white" if value > thresh else "black"
            plt.text(j, i, format(value, fmt),
                     horizontalalignment="center",
                     color=color)
        # Case 2: Last column (but not the very bottom-right corner):
        elif j == n and i < n:
            # Absolute total for row i
            value = row_totals[i]
            plt.text(j, i, str(value),
                     horizontalalignment="center",
                     color="black")
        # Case 3: Last row (but not the very bottom-right corner):
        elif i == n and j < n:
            # Absolute total for column j
            value = col_totals[j]
            plt.text(j, i, str(value),
                     horizontalalignment="center",
                     color="black")
        # Case 4: Bottom-right corner cell
        else:
            # Grand total
            value = grand_total
            plt.text(j, i, str(value),
                     horizontalalignment="center",
                     color="black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(savepath, bbox_inches='tight')
    plt.close()


def load_consensus_predictions(pred_csv_path1, pred_csv_path2, label_column_names):
    """
    Loads predictions from two CSV files and returns only predictions where
    both files agree on the Result Number.

    Parameters:
        pred_csv_path1 (str): Path to first prediction CSV
        pred_csv_path2 (str): Path to second prediction CSV
        label_column_names (list): List of column names containing the predictions

    Returns:
        dict: Filename -> class_id mapping for samples where both predictions agree
    """
    # Load both prediction files
    pred_map1 = {}
    pred_map2 = {}
    
    # Load first prediction file
    with open(pred_csv_path1, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            original_path = row['Original Image Path'].strip()
            filename_only = os.path.basename(original_path)

            if len(label_column_names) < 2:
                result_number = row[label_column_names[0]]
                
                # Skip rows with empty Result Number
                if result_number.strip() == '':
                    continue

                predicted_class_id = int(row[label_column_names[0]])
            else:
                predicted_class_id = voting_from_row(row, label_column_names)

            pred_map1[filename_only] = predicted_class_id
    
    # Load second prediction file
    with open(pred_csv_path2, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            original_path = row['Original Image Path'].strip()
            filename_only = os.path.basename(original_path)

            if len(label_column_names) < 2:
                result_number = row[label_column_names[0]]
                
                # Skip rows with empty Result Number
                if result_number.strip() == '':
                    continue

                predicted_class_id = int(row[label_column_names[0]])
            else:
                predicted_class_id = voting_from_row(row, label_column_names)

            pred_map2[filename_only] = predicted_class_id

    # Create consensus map
    consensus_pred_map = {}
    
    # Find common filenames
    common_files = set(pred_map1.keys()) & set(pred_map2.keys())
    
    # Keep only predictions that match
    for filename in common_files:
        if pred_map1[filename] == pred_map2[filename]:
            consensus_pred_map[filename] = pred_map1[filename]
    
    # Print some statistics
    print(f"Total predictions in first file: {len(pred_map1)}")
    print(f"Total predictions in second file: {len(pred_map2)}")
    print(f"Common files: {len(common_files)}")
    print(f"Matching predictions: {len(consensus_pred_map)}")
    print(f"Agreement rate: {100 * len(consensus_pred_map) / len(common_files):.2f}%")
    
    return consensus_pred_map

# ---------------------------------------------------------------------------
# 4. Main Script
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate predictions against ground truth.")
    parser.add_argument('--gt_csv',   type=str, required=True, help="Path to ground truth CSV.")
    parser.add_argument('--pred_csv', type=str, required=True, help="Path to predictions CSV.")
    parser.add_argument('--out_cm',   type=str, default='confusion_matrix.png',
                        help="Output path for the confusion matrix figure.")
    parser.add_argument('--crops', action='store_true',
                        help="If set, all classes >= 9 are mapped to class 9.")
    return parser.parse_args()

def main():
    # args = parse_args()

    gt_csv = '/home/laguarta_jordi/sean7391/streetview_highres/vlms/labels1/cropLabelsBatch1Processed.csv'
    # pred_csv = '/home/laguarta_jordi/sean7391/streetview_highres/vlms/labelsBatch1/claude/kharif_2023_1k_results.csv'
    pred_csv = '/home/laguarta_jordi/sean7391/streetview_highres/vlms/labelsBatch1/gpt4-1votes-bClasses-longPrompt-finetuned/kharif_2023_1k_results.csv'
    # pred_csv1 = '/home/laguarta_jordi/sean7391/streetview_highres/vlms/labelsBatch1/gpt4/kharif_2023_1k_results.csv'
    out_folder = '/home/laguarta_jordi/sean7391/streetview_highres/vlms/labelsBatch1/gpt4-1votes-bClasses-longPrompt-finetuned/'
    
    crops = True
    high_conf = True
    mid_conf = False
    collapse_legumes = True
    label_column_names = ['Result Number 1']

    title = 'Gemini LongPrompt Finetuned'
    
    
    # 1. Load data
    gt_map   = load_ground_truth(gt_csv, high_conf, mid_conf)
    pred_map = load_predictions(pred_csv, label_column_names)
    # pred_map = load_consensus_predictions(pred_csv, pred_csv1, label_column_names)
    
    
    # 2. Align data
    y_true, y_pred, matched_files = evaluate_predictions(gt_map, pred_map)

    if not y_true:
        print("No overlapping filenames between ground truth and predictions. Nothing to evaluate.")
        return

    # 3. If --crops is set, collapse classes >= 9 to 9
    if crops:
        y_true = [min(c, 9) for c in y_true]
        y_pred = [min(c, 9) for c in y_pred]

    if collapse_legumes:
        y_true = [1 if c == 6 else c for c in y_true]
        y_pred = [1 if c == 6 else c for c in y_pred]


    # 4. Compute metrics
    # Accuracy
    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {acc:.4f}")

    # Decide which classes exist overall.
    # If crops is set, we effectively have classes 0..9; otherwise 0..14
    if crops:
        used_labels = list(range(10))  # classes 0..9
    else:
        used_labels = list(range(15))  # classes 0..14

    # Confusion Matrix (absolute)
    cm_abs = confusion_matrix(y_true, y_pred, labels=used_labels)
    print("Confusion Matrix (Absolute):")
    print(cm_abs)

    # Confusion Matrix (row-wise percentage)
    cm_sum = cm_abs.sum(axis=1, keepdims=True)
    cm_perc = cm_abs / (cm_sum + 1e-9) * 100.0
    print("Confusion Matrix (% by row):")
    with np.printoptions(precision=2, suppress=True):
        print(cm_perc)

    # Classification report using the string labels
    # We'll create target_names from the CLASS_NAME_MAP for the used labels
    target_names = [CLASS_NAME_MAP[i] for i in used_labels]
    report = classification_report(
        y_true, 
        y_pred, 
        labels=used_labels, 
        target_names=target_names, 
        zero_division=0
    )
    print("Classification Report:")
    print(report)

    extras = ''
    if high_conf:
        extras+= '_highconf'
    elif mid_conf:
        extras+= '_midconf'
    if crops:
        extras += '_crops'
    if collapse_legumes:
        extras += '_legumes'
    # 5. Save Confusion Matrix Figure
    plot_confusion_matrix(
        cm_abs,
        class_names=target_names,
        savepath=f"{out_folder}confusion_matrix_{extras}",
        title=(f"{title} ({round(acc,2)})"),
        percent=True
    )
    print(f"Saved confusion matrix figure")


if __name__ == "__main__":
    main()
