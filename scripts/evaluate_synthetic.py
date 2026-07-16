"""Evaluate the model on the synthetic dataset."""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from omegaconf import OmegaConf
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_recall_curve,
    f1_score,
    average_precision_score,
    auc,
    r2_score,
    mean_absolute_error,
    classification_report,
)
from sklearn.preprocessing import label_binarize
from moldetr import plotstyle
from moldetr.config import MultipletConfig, resolve_device
from moldetr.dataloader.transforms import Transform
from moldetr.reproducibility import set_seed
from moldetr.learner.multi_multiplet_learner import init_learner
from moldetr.loss.helper_funcs import get_src_permutation_idx
from scipy.special import softmax
from hydra.core.config_store import ConfigStore
import hydra
import matplotlib

# Define constants
NO_SPIN_CLASS = 5  # Adjust this based on your specific class index for 'no spin'

# Apply the shared MolDeTr publication style (palette, fonts, 300-dpi saves), then use the
# categorical spin-system colours as the default cycle. This script lays out with tight_layout
# and seaborn, so constrained_layout is turned back off to avoid layout-engine conflicts.
plotstyle.apply_style()
plt.rcParams["figure.constrained_layout.use"] = False
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=plotstyle.SPIN_SYSTEM_COLORS)

# Two categorical Okabe-Ito colours for the target-vs-prediction overlays (colourblind-safe).
TRUTH_C = plotstyle.SPIN_SYSTEM_COLORS[1]  # targets / ground truth — sky blue
PRED_C = plotstyle.SPIN_SYSTEM_COLORS[4]  # predictions — vermillion


def get_class(class_predictions: list[np.ndarray], threshold: float = 0.0) -> list[tuple[int, int]]:
    """
    Determine the predicted class of the multiplets in the batch sample.

    Parameters
    ----------
    class_predictions: list of np.ndarray
        Predicted class probabilities for the multiplets in the batch sample.
    threshold: float
        Threshold for class prediction.

    Returns
    -------
    list of tuples (index, class)
        Contains the index of the prediction and the predicted class of the multiplet.
    """
    multiplet_list = []
    for index, class_prediction in enumerate(class_predictions):
        if np.any(class_prediction > threshold):
            multiplet_list.append((index, np.argmax(class_prediction)))
    return multiplet_list


def unnorm(
    normed_predictions: torch.Tensor, normed_targets: torch.Tensor, transform: Transform
) -> (list, list):
    """
    Unnormalize the predictions and targets to the original scale of the dataset.

    Parameters
    ----------
    normed_predictions: torch.Tensor
        Tensor containing the normalized predictions of the model.
    normed_targets: torch.Tensor
        Tensor containing the normalized targets of the model.
    transform: Transform
        Object containing the data set and the normalization parameters.

    Returns
    -------
    tuple of lists
        Unnormalized predictions and targets.
    """
    num_params = len(transform.extrema)
    unnormed_predictions = []
    unnormed_targets = []

    for normed_prediction in normed_predictions:
        unnormed_pred = [
            transform.untransform(float(normed_prediction[param_index]), param_index)
            for param_index in range(min(num_params, normed_prediction.shape[0]))
        ]
        unnormed_predictions.append(unnormed_pred)

    for normed_target in normed_targets:
        unnormed_tgt = [
            transform.untransform(float(normed_target[param_index]), param_index)
            for param_index in range(min(num_params, normed_target.shape[0]))
        ]
        unnormed_targets.append(unnormed_tgt)

    return unnormed_predictions, unnormed_targets


def plot_prediction(
    index: int,
    sample: np.ndarray,
    param_predictions: list,
    param_targets: list,
    class_targets: np.ndarray,
    reg_param_indices: dict,
    multiplet_list: list,
    points_per_ppm: float,
    class_names: dict,
):
    """
    Plot predictions and targets for a given sample with visual indications of their positions and bounding boxes.

    Parameters
    ----------
    index: int
        Index of the sample.
    sample: np.ndarray
        The sample data.
    param_predictions: list
        List of parameter predictions.
    param_targets: list
        List of parameter targets.
    class_targets: np.ndarray
        Array of class targets.
    reg_param_indices: dict
        Dictionary of regression parameter indices.
    multiplet_list: list
        List of multiplets.
    points_per_ppm: float
        Number of points per ppm.
    class_names: dict
        Dictionary of class names.
    """
    if sample is None or not hasattr(sample, "shape") or sample.shape == ():
        print(f"Warning: Sample at index {index} is empty or has an invalid shape. Skipping.")
        return

    if len(sample.shape) == 1:
        sample = sample.reshape(1, -1)
    elif len(sample.shape) < 2:
        raise ValueError(f"Sample at index {index} has an unexpected shape: {sample.shape}")

    if sample.shape[0] == 1:
        sample = sample.squeeze(0)

    fig, ax = plt.subplots(figsize=(12, 4))
    x_points = np.arange(sample.shape[-1])
    x_ppm = x_points / points_per_ppm

    ax.plot(x_ppm, sample, linewidth=2, color=plotstyle.SPECTRUM_COLOR, label="NMR Spectrum")

    for i, (c, target) in enumerate(zip(class_targets, param_targets)):
        try:
            target_center = target[reg_param_indices["center_position_in_points"]] / points_per_ppm
            target_range = (
                target[reg_param_indices["bounding_box_range_in_points"]] / points_per_ppm
            )

            ax.axvline(target_center, color=TRUTH_C, linestyle="--", alpha=0.6)
            ax.axvspan(
                target_center - target_range / 2,
                target_center + target_range / 2,
                color=TRUTH_C,
                alpha=0.3,
                label="Spin target" if i == 0 else None,
            )
            ax.text(
                target_center - target_range / 2 - 0.01,
                0.8,
                f"{int(c + 1)}p",
                fontsize=12,
                fontweight="bold",
                color=TRUTH_C,
            )
        except IndexError as e:
            print(f"Error plotting target at index {i} for sample {index}: {e}")
            continue

    for i, (m, c) in enumerate(multiplet_list):
        try:
            prediction = param_predictions[m]
            pred_center = (
                prediction[reg_param_indices["center_position_in_points"]] / points_per_ppm
            )
            pred_range = (
                prediction[reg_param_indices["bounding_box_range_in_points"]] / points_per_ppm
            )

            ax.axvline(pred_center, color=PRED_C, linestyle="--", alpha=0.6)
            ax.axvspan(
                pred_center - pred_range / 2,
                pred_center + pred_range / 2,
                color=PRED_C,
                alpha=0.3,
                label="Spin prediction" if i == 0 else None,
            )
            ax.text(
                pred_center - pred_range / 2 - 0.01,
                0.6,
                f"{int(c + 1)}p",
                fontsize=12,
                fontweight="bold",
                color=PRED_C,
            )
        except IndexError as e:
            print(f"Error plotting prediction at index {i} for sample {index}: {e}")
            continue

    ax.set_xlabel("Chemical shift (ppm)", fontsize=14)
    ax.set_ylabel("Intensity (a.u.)", fontsize=14)
    ax.legend(fontsize=12, loc="upper right")

    ax.invert_xaxis()
    plt.tight_layout()
    plt.show()


def sort_according_to_matching(
    outputs: torch.Tensor,
    targets: dict,
    indices: list,
) -> (torch.Tensor, torch.Tensor):
    """
    Sort outputs and targets based on matching indices.

    Parameters
    ----------
    outputs: torch.Tensor
        Model outputs.
    targets: dict
        Target dictionary containing 'targets'.
    indices: list
        List of matching indices.

    Returns
    -------
    tuple
        Sorted predictions and sorted targets.
    """
    output_index = get_src_permutation_idx(indices)
    targets_permuted = torch.cat([t[J] for t, (_, J) in zip(targets["targets"], indices)])
    prediction_permuted = outputs[output_index]
    return prediction_permuted, targets_permuted


def points_to_ppm(points, npoints, base_freq_in_MHz, max_ppm, min_ppm):
    """
    Convert points to ppm.

    Parameters
    ----------
    points: np.ndarray
        Data points.
    npoints: int
        Number of points.
    base_freq_in_MHz: float
        Base frequency in MHz.
    max_ppm: float
        Maximum ppm value.
    min_ppm: float
        Minimum ppm value.

    Returns
    -------
    np.ndarray
        ppm values.
    """
    ppm = (points / (npoints - 1)) * (max_ppm - min_ppm) + min_ppm
    return ppm


def points_to_hz(points, npoints, base_freq_in_MHz, max_ppm, min_ppm):
    """
    Convert points to Hz.

    Parameters
    ----------
    points: np.ndarray
        Data points.
    npoints: int
        Number of points.
    base_freq_in_MHz: float
        Base frequency in MHz.
    max_ppm: float
        Maximum ppm value.
    min_ppm: float
        Minimum ppm value.

    Returns
    -------
    np.ndarray
        Hz values.
    """
    hz = (points / (npoints - 1)) * (base_freq_in_MHz * (max_ppm - min_ppm))
    return hz


def plot_predictions_vs_truth(
    predictions,
    targets,
    param_name,
    scale="ppm",
    npoints=None,
    base_freq_in_MHz=None,
    max_ppm=None,
    min_ppm=None,
):
    """
    Plot predictions vs. ground truth and residuals using sns.jointplot,
    with styling adapted to match high-quality publications.

    Parameters
    ----------
    predictions: np.ndarray
        Predicted values.
    targets: np.ndarray
        Ground truth values.
    param_name: str
        Name of the parameter (e.g., 'Chemical Shift', 'Coupling Constant').
    scale: str, optional
        Scale to use ('ppm' or 'hz').
    npoints: int, optional
        Number of points (required if scale conversion is needed).
    base_freq_in_MHz: float, optional
        Base frequency in MHz (required if scale conversion is needed).
    max_ppm: float, optional
        Maximum ppm value (required if scale conversion is needed).
    min_ppm: float, optional
        Minimum ppm value (required if scale conversion is needed).
    """

    # Convert predictions and targets to the specified scale
    if scale == "ppm":
        predictions = points_to_ppm(predictions, npoints, base_freq_in_MHz, max_ppm, min_ppm)
        targets = points_to_ppm(targets, npoints, base_freq_in_MHz, max_ppm, min_ppm)
        ylabel = "Predicted (ppm)"
        xlabel = f"True {param_name} (ppm)"
        ylabel_r = "Residuals (ppm)"
    elif scale == "hz":
        predictions = points_to_hz(predictions, npoints, base_freq_in_MHz, max_ppm, min_ppm)
        targets = points_to_hz(targets, npoints, base_freq_in_MHz, max_ppm, min_ppm)
        ylabel = "Predicted (Hz)"
        xlabel = f"True {param_name} (Hz)"
        ylabel_r = "Residuals (Hz)"
    else:
        ylabel = f"Predicted {param_name}"
        xlabel = f"True {param_name}"
        ylabel_r = f"Residuals {param_name}"

    # Calculating residuals
    residuals = predictions - targets

    # Determine the limits based on the parameter name
    if param_name == "Chemical Shift":
        x_lim = (0, 1200)
        y_lim = (0, 1200)
    elif param_name == "Coupling Constant":
        x_lim = (0.0, 20.3)
        y_lim = (0.0, 20.3)
    else:
        x_min = min(targets)
        y_min = min(predictions)
        x_max = max(targets)
        y_max = max(predictions)
        x_lim = (max(x_min - (x_max - x_min) * 0.05, 0), x_max + (x_max - x_min) * 0.05)
        y_lim = (max(y_min - (y_max - y_min) * 0.05, 0), y_max + (y_max - y_min) * 0.05)

    # Two categorical Okabe-Ito colours: data points vs. the regression line / 95% CI.
    base_color = plotstyle.SPIN_SYSTEM_COLORS[1]  # data points — sky blue
    reg_color = plotstyle.SPIN_SYSTEM_COLORS[4]  # regression line + CI — vermillion

    # Initialize Seaborn style
    sns.set(style="whitegrid")
    sns.set_style("ticks")

    # Plot Prediction vs. Ground Truth
    g1 = sns.jointplot(
        x=targets,
        y=predictions,
        kind="reg",
        height=6,
        ratio=4,
        space=0.1,
        marginal_ticks=True,
        color=base_color,
        marginal_kws={"bins": 20, "fill": True, "color": base_color, "stat": "count"},
        line_kws={"color": reg_color, "linewidth": 2},
        scatter_kws={"alpha": 0.4},
    )

    g1.set_axis_labels(xlabel, ylabel, fontsize=14)

    # Adjust x and y limits
    if param_name in ["Chemical Shift", "Coupling Constant"]:
        g1.ax_joint.set_xlim(x_lim)
        g1.ax_joint.set_ylim(y_lim)
    else:
        # Ensure lower limits are at least zero
        current_x_min, current_x_max = g1.ax_joint.get_xlim()
        current_y_min, current_y_max = g1.ax_joint.get_ylim()

        g1.ax_joint.set_xlim(max(x_lim[0], 0), x_lim[1])
        g1.ax_joint.set_ylim(max(y_lim[0], 0), y_lim[1])

    # Decrease the transparency of the 95% CI area
    ci_poly = [
        child
        for child in g1.ax_joint.get_children()
        if isinstance(child, matplotlib.collections.PolyCollection)
    ]
    for poly in ci_poly:
        poly.set_alpha(0.5)

    # Add grid lines
    g1.ax_joint.grid(True)
    g1.ax_marg_x.grid(True)
    g1.ax_marg_y.grid(True)

    # Plot the ideal y = x line
    g1.ax_joint.plot(
        [x_lim[0], x_lim[1]],
        [y_lim[0], y_lim[1]],
        color="black",
        linestyle="--",
        label="Ideal (y = x)",
    )

    # Add custom lines to the legend
    reg_line = plt.Line2D([], [], color=reg_color, linewidth=2, label="Fit ± 95% CI")
    ideal_line = plt.Line2D([], [], color="black", linestyle="--", label="Ideal (y = x)")
    g1.ax_joint.legend(handles=[reg_line, ideal_line], fontsize=12)

    # Adjust tick label sizes
    g1.ax_joint.tick_params(axis="both", which="major", labelsize=12)
    g1.ax_marg_x.tick_params(axis="x", which="major", labelsize=12)
    g1.ax_marg_y.tick_params(axis="y", which="major", labelsize=12)

    plt.tight_layout()
    plt.gcf().set_dpi(300)
    plt.show()

    # Plot Residuals vs. Ground Truth
    g2 = sns.jointplot(
        x=targets,
        y=residuals,
        kind="reg",
        height=6,
        ratio=4,
        space=0.1,
        color=base_color,
        marginal_ticks=True,
        marginal_kws={"bins": 20, "fill": True, "color": base_color, "stat": "count"},
        line_kws={"color": reg_color, "linewidth": 2},
        scatter_kws={"alpha": 0.4},
    )

    g2.set_axis_labels(xlabel, ylabel_r, fontsize=14)

    # Decrease the transparency of the 95% CI area
    ci_poly = [
        child
        for child in g2.ax_joint.get_children()
        if isinstance(child, matplotlib.collections.PolyCollection)
    ]
    for poly in ci_poly:
        poly.set_alpha(0.5)

    # Add horizontal line at y=0
    g2.ax_joint.axhline(0, color="black", linestyle="--", label="Zero Reference")
    g2.ax_joint.grid(True)
    g2.ax_marg_x.grid(True)
    g2.ax_marg_y.grid(True)

    # Add custom lines to the legend
    reg_line_residuals = plt.Line2D([], [], color=reg_color, linewidth=2, label="Fit ± 95% CI")
    zero_ref_line = plt.Line2D([], [], color="black", linestyle="--", label="Zero Reference")
    g2.ax_joint.legend(handles=[reg_line_residuals, zero_ref_line], fontsize=12)

    # Adjust x and y limits
    if param_name in ["Chemical Shift", "Coupling Constant"]:
        g2.ax_joint.set_xlim(x_lim)
        # For residuals, center around zero with padding
        residual_range = max(abs(residuals.min()), abs(residuals.max()))
        residual_buffer = residual_range * 0.05
        g2.ax_joint.set_ylim(-residual_range - residual_buffer, residual_range + residual_buffer)
    else:
        # Ensure x lower limit is at least zero
        current_x_min_res, current_x_max_res = g2.ax_joint.get_xlim()
        g2.ax_joint.set_xlim(max(x_lim[0], 0), x_lim[1])

        residual_buffer = (max(residuals) - min(residuals)) * 0.05
        g2.ax_joint.set_ylim(min(residuals) - residual_buffer, max(residuals) + residual_buffer)

    # Adjust tick label sizes
    g2.ax_joint.tick_params(axis="both", which="major", labelsize=12)
    g2.ax_marg_x.tick_params(axis="x", which="major", labelsize=12)
    g2.ax_marg_y.tick_params(axis="y", which="major", labelsize=12)

    plt.tight_layout()
    plt.gcf().set_dpi(300)
    plt.show()


def calculate_iou_1d(interval1, interval2):
    """
    Calculate the Intersection over Union (IoU) for 1D intervals.

    Parameters
    ----------
    interval1: tuple
        First interval (start, end).
    interval2: tuple
        Second interval (start, end).

    Returns
    -------
    float
        IoU value.
    """
    a_start, a_end = interval1
    b_start, b_end = interval2
    overlap = max(0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    return overlap / union if union != 0 else 0


def center_width_to_interval(center, width):
    """
    Convert center and width to an interval.

    Parameters
    ----------
    center: float
        Center value.
    width: float
        Width value.

    Returns
    -------
    tuple
        Interval (start, end).
    """
    return (center - width / 2, center + width / 2)


def calculate_ap_for_1d_detections(matched_gt_preds, overlap_thresholds):
    """
    Calculate Average Precision (AP) for 1D detections and plot AP vs. Overlap Thresholds.

    Parameters
    ----------
    matched_gt_preds: list
        List of matched ground truths and predictions.
    overlap_thresholds: list
        List of overlap thresholds.

    Returns
    -------
    dict
        Average precisions for different overlap thresholds.
    """
    matched_intervals = [
        (
            center_width_to_interval(gt_center, gt_width),
            center_width_to_interval(pred_center, pred_width),
            confidence,
        )
        for gt_center, gt_width, pred_center, pred_width, confidence in matched_gt_preds
    ]

    # Sort by confidence in descending order
    matched_intervals.sort(key=lambda x: x[2], reverse=True)

    average_precisions = {}

    for overlap_threshold in overlap_thresholds:
        tp_fp_labels = []
        confidences = []

        for gt_interval, pred_interval, confidence in matched_intervals:
            iou = calculate_iou_1d(gt_interval, pred_interval)
            is_true_positive = iou >= overlap_threshold
            tp_fp_labels.append(is_true_positive)
            confidences.append(confidence)

        # Calculate precision and recall
        precisions, recalls, _ = precision_recall_curve(tp_fp_labels, confidences)

        # Calculate AP using AUC
        ap = auc(recalls, precisions)
        average_precisions[overlap_threshold] = ap

    # Plotting Average Precision vs. Overlap Thresholds
    thresholds = list(average_precisions.keys())
    ap_values = list(average_precisions.values())

    plt.figure(figsize=(6, 4), dpi=300)
    plt.plot(
        thresholds,
        ap_values,
        marker="o",
        linestyle="-",
        color="k",
        markersize=6,
        linewidth=2,
    )
    plt.xlabel("Overlap Thresholds", fontsize=14)
    plt.ylabel("Average Precision", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(False)
    plt.tight_layout()
    plt.show()

    return average_precisions


def plot_confusion_matrix(y_true, y_pred, labels_dict, title="Confusion Matrix", alpha=0.7):
    """
    Creates a confusion matrix plot based on the true and predicted labels,
    styled to match high-quality publication standards.

    Parameters
    ----------
    y_true: array-like
        True labels of the samples.
    y_pred: array-like
        Predicted labels of the samples.
    labels_dict: dict
        Dictionary mapping class names to their integer indices.
    title: str, optional
        Title of the plot.
    alpha: float, optional
        Transparency level between 0 (transparent) and 1 (opaque).
    """
    # Extract the labels from the dictionary
    labels = [key for key in labels_dict if labels_dict[key] in np.unique(y_true)]

    # Map the true and predicted labels to the corresponding names using the dictionary
    y_true_named = np.array(
        [key for value in y_true for key, val in labels_dict.items() if val == value]
    )
    y_pred_named = np.array(
        [key for value in y_pred for key, val in labels_dict.items() if val == value]
    )

    # Calculate the confusion matrix using the named labels
    cm = confusion_matrix(y_true_named, y_pred_named, labels=labels)

    # Normalize the confusion matrix to proportions
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    # Create a transparent version of a perceptual colormap (viridis; genuine continuous data).
    cmap = matplotlib.colormaps["viridis"]
    colors = cmap(np.linspace(0, 1, cmap.N))
    colors[:, -1] = alpha
    transparent_cmap = matplotlib.colors.ListedColormap(colors)

    # Create a figure and a set of subplots
    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)

    # Turn off the axes grid lines
    ax.grid(False)

    # Remove the axes spines for a cleaner look
    sns.despine(ax=ax, top=True, right=True, left=True, bottom=True)

    # Create a heatmap without lines and with larger annotations
    heatmap = sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2%",
        cmap=transparent_cmap,
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={"label": "Proportion of Predictions"},
        linewidths=0,
        linecolor="white",
        ax=ax,
        annot_kws={"size": 20},
    )

    # Set axis labels and title with larger font sizes
    ax.set_xlabel("Predicted Proton Count", fontsize=20)
    ax.set_ylabel("True Proton Count", fontsize=20)

    # Adjust tick label size
    ax.tick_params(axis="both", which="major", labelsize=18)

    # Manually set the font size for the color bar
    colorbar = heatmap.collections[0].colorbar
    colorbar.ax.tick_params(labelsize=18)
    colorbar.set_label("Proportion of Predictions", fontsize=18)

    # Adjust the layout to prevent clipping of labels
    plt.tight_layout()

    # Show the plot
    plt.show()


def precision_recall_curve_plot(
    class_prediction_list: list[np.ndarray],
    class_target_list: list[np.ndarray],
    label_dict: dict,
    threshold_increment: float = 0.05,
    apply_sigmoid: bool = False,
):
    """
    Plots precision-recall curves, F1 score vs. threshold, and average precision vs. threshold
    for each class in a multi-class classification problem.

    Parameters
    ----------
    class_prediction_list : list of np.ndarray
        List containing arrays of class probability predictions or logits for each sample.
    class_target_list : list of np.ndarray
        List containing arrays of true class labels for each sample.
    label_dict : dict
        Dictionary mapping class names to their integer indices.
    threshold_increment : float, optional
        Increment for threshold values used to find the best threshold, by default 0.05.
    apply_sigmoid : bool, optional
        Whether to apply the sigmoid function to the predictions (useful if predictions are logits), by default False.

    Returns
    -------
    dict
        Best thresholds for each class based on F1 score.
    """
    # Concatenate all predictions and targets
    class_predictions_flat = np.concatenate(class_prediction_list, axis=0)
    class_targets_flat = np.concatenate(class_target_list, axis=0)

    # Apply sigmoid if predictions are logits
    if apply_sigmoid:
        class_predictions_flat = 1 / (1 + np.exp(-class_predictions_flat))

    num_classes = class_predictions_flat.shape[1]

    # Binarize targets for multi-class handling
    class_target_one_hot = label_binarize(class_targets_flat, classes=range(num_classes))

    # Initialize dictionaries to store metrics
    precision = {}
    recall = {}
    f1_scores = {}
    thresholds = {}

    # Per-class categorical colours (colourblind-safe Okabe-Ito).
    colors = plotstyle.SPIN_SYSTEM_COLORS

    # Plot Precision-Recall curves for each class
    plt.figure(figsize=(6, 4), dpi=300)
    for class_idx in range(num_classes):
        precision[class_idx], recall[class_idx], thresholds[class_idx] = precision_recall_curve(
            class_target_one_hot[:, class_idx], class_predictions_flat[:, class_idx]
        )
        f1_scores[class_idx] = (
            2
            * (precision[class_idx] * recall[class_idx])
            / (precision[class_idx] + recall[class_idx] + 1e-7)
        )

        # Compute AUC for the Precision-Recall curve
        pr_auc = auc(recall[class_idx], precision[class_idx])

        # Plot the precision-recall curve for each class
        label = [key for key, value in label_dict.items() if value == class_idx][0]
        plt.plot(
            recall[class_idx],
            precision[class_idx],
            label=f"{label} (AUC={pr_auc:.2f})",
            color=colors[class_idx],
            linewidth=2,
            alpha=0.6,
        )

    plt.xlabel("Recall", fontsize=14)
    plt.ylabel("Precision", fontsize=14)
    plt.legend(fontsize=10)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(False)
    plt.tight_layout()
    plt.show()

    # Find the threshold that maximizes the F1 score for each class
    threshold_values = np.linspace(0, 1, int(1 / threshold_increment) + 1)
    best_thresholds = {}

    plt.figure(figsize=(6, 4), dpi=300)
    for class_idx in range(num_classes):
        class_predictions = class_predictions_flat[:, class_idx]
        f1_scores_for_thresholds = np.array(
            [
                f1_score(
                    class_target_one_hot[:, class_idx],
                    (class_predictions >= threshold).astype(int),
                )
                if ((class_predictions >= threshold).sum() > 0)
                else 0
                for threshold in threshold_values
            ]
        )

        label = [key for key, value in label_dict.items() if value == class_idx][0]
        plt.plot(
            threshold_values,
            f1_scores_for_thresholds,
            label=label,
            color=colors[class_idx],
            linewidth=2,
            alpha=0.6,
        )

        # Find the threshold that maximizes the F1 score
        best_threshold_index = np.argmax(f1_scores_for_thresholds)
        best_threshold = threshold_values[best_threshold_index]
        best_thresholds[class_idx] = best_threshold
        plt.scatter(
            best_threshold,
            f1_scores_for_thresholds[best_threshold_index],
            color=colors[class_idx],
            edgecolor="black",
            zorder=5,
        )

    plt.xlabel("Threshold", fontsize=14)
    plt.ylabel("F1 Score", fontsize=14)
    plt.legend(fontsize=10)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(False)
    plt.tight_layout()
    plt.show()

    # Plot Average Precision vs. Threshold for each class
    plt.figure(figsize=(6, 4), dpi=300)
    for class_idx in range(num_classes):
        label = [key for key, value in label_dict.items() if value == class_idx][0]
        average_precisions = []
        class_predictions = class_predictions_flat[:, class_idx]

        for threshold in threshold_values:
            preds_binarized = (class_predictions >= threshold).astype(int)
            if preds_binarized.sum() == 0:
                average_precisions.append(0)
                continue
            average_precisions.append(
                average_precision_score(class_target_one_hot[:, class_idx], preds_binarized)
            )

        plt.plot(
            threshold_values,
            average_precisions,
            label=label,
            color=colors[class_idx],
            linewidth=2,
            alpha=0.6,
        )

    plt.xlabel("Threshold", fontsize=14)
    plt.ylabel("Average Precision", fontsize=14)
    plt.legend(fontsize=10)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(False)
    plt.tight_layout()
    plt.show()

    return best_thresholds


# Create config object for the model and training process and store it in the config store
cs = ConfigStore.instance()
cs.store(name="multiplet_config", node=MultipletConfig)


@hydra.main(config_path="../conf", config_name="config_big")
def main(cfg: MultipletConfig) -> None:
    """
    Main function to evaluate a model on a synthetic dataset and plot the results.

    Parameters
    ----------
    cfg: MultipletConfig
        Configuration object containing all parameters for the model and training process,
        as well as the paths to the model and synthetic dataset.
    """
    # Initialize configuration
    cfg = OmegaConf.structured(cfg)
    print(OmegaConf.to_yaml(cfg))

    # Initialize the learner
    set_seed(42)
    learner = init_learner(cfg, test=True)

    # Set device

    device = resolve_device(cfg.device.device_name)
    print(f"Using device: {device}")

    # Load the model
    learner.load(cfg.lognames.best_model_file)
    learner.model.eval().to(device)

    # Ensure DataLoader is set to the correct device
    learner.dls[0].device = device

    NO_SPIN_CLASS = 5  # Adjust this based on your specific class index for 'no spin'

    # Define a threshold for high cost
    high_cost_threshold = torch.tensor(0.5, device=device)

    # We'll only use the first group
    n_groups = cfg.optim_params.n_groups

    # Initialize lists to collect unmatched predictions and targets
    never_matched_predictions = []
    unmatched_predictions = []
    unmatched_targets = []
    error_list = []

    # Inference loop
    for batch_index, batch in enumerate(learner.dls[0]):
        with torch.no_grad():
            batch_sample, batch_target = batch
            batch_sample = batch_sample.to(device)

            # Get model predictions
            batch_prediction = learner.model(batch_sample)

            # Compute the error for each sample in the batch
            error_tensor, group_indices, group_costs = learner.loss_func(
                batch_prediction, batch_target, reduction="none"
            )

            # Sanity check: Ensure error_tensor size matches expectations
            batch_size = batch_sample.size(0)
            expected_size = n_groups * batch_size
            actual_size = error_tensor.numel()
            assert actual_size == expected_size, (
                f"Expected error_tensor size {expected_size}, got {actual_size}"
            )
            error_tensor = error_tensor.view(n_groups, batch_size)

            # Process only the first group
            group_output = batch_prediction[
                :, 0
            ]  # Shape: (batch_size, num_queries, num_params + num_classes)
            group_matching = group_indices[0]  # Assuming list of tuples
            group_cost = group_costs[0]["cost_param"].to(
                device
            )  # Shape: (batch_size * num_queries, total_num_targets)

            num_queries = group_output.shape[1]
            adjusted_group_matching = []

            for sample_idx in range(batch_size):
                pred_indices, tgt_indices = group_matching[sample_idx]

                never_matched_pred_indices = [
                    i for i in range(num_queries) if i not in pred_indices
                ]
                never_matched_predictions.extend(
                    group_output[sample_idx, never_matched_pred_indices].cpu().numpy()
                )

                # Ensure indices are tensors on the correct device
                pred_indices = torch.as_tensor(pred_indices, device=device)
                tgt_indices = torch.as_tensor(tgt_indices, device=device)

                # Calculate prediction indices for this sample
                pred_start = sample_idx * num_queries
                pred_end = (sample_idx + 1) * num_queries

                # Get target start and end indices using cumulative sum
                num_targets_list = batch_target["num_targets"]
                num_targets_tensor = torch.tensor(num_targets_list, device=device)
                tgt_cumsum = torch.cat(
                    (
                        torch.tensor([0], device=device),
                        torch.cumsum(num_targets_tensor, dim=0),
                    )
                )

                tgt_start = tgt_cumsum[sample_idx].item()
                tgt_end = tgt_cumsum[sample_idx + 1].item()

                # Get sample cost matrix
                sample_cost = group_cost[pred_start:pred_end, tgt_start:tgt_end]

                # Index sample_cost with pred_indices and tgt_indices
                if pred_indices.numel() == 0 or tgt_indices.numel() == 0:
                    print(f"Sample {sample_idx} has empty prediction or target indices.")
                    continue

                try:
                    matched_costs = sample_cost[pred_indices, tgt_indices]
                except IndexError as e:
                    print(f"Batch {batch_index} - Sample {sample_idx}: IndexError - {e}")
                    continue

                # Identify high-cost matches
                high_cost_mask = matched_costs > high_cost_threshold

                # Separate high-cost and low-cost matches
                pred_indices_high_cost = pred_indices[high_cost_mask]
                tgt_indices_high_cost = tgt_indices[high_cost_mask]

                pred_indices_low_cost = pred_indices[~high_cost_mask]
                tgt_indices_low_cost = tgt_indices[~high_cost_mask]

                # Collect unmatched predictions and targets from high-cost matches
                if pred_indices_high_cost.numel() > 0:
                    # Extract the actual prediction vectors
                    unmatched_preds = group_output[sample_idx, pred_indices_high_cost].cpu().numpy()
                    unmatched_predictions.extend(unmatched_preds)

                    # Extract the actual target vectors
                    sample_unmatched_tgt_indices = tgt_indices_high_cost
                    unmatched_tgts = batch_target["targets"][sample_idx][
                        sample_unmatched_tgt_indices.cpu().numpy()
                    ]
                    unmatched_targets.extend(unmatched_tgts)

                # Update the matching indices with low-cost matches
                adjusted_group_matching.append((pred_indices_low_cost, tgt_indices_low_cost))

            # Sort predictions and targets according to adjusted matching
            batch_prediction_sorted, batch_target_sorted = sort_according_to_matching(
                group_output, batch_target, adjusted_group_matching
            )

            # Get the error tensor for the first group
            group_error = error_tensor[0]  # Shape: (batch_size,)

            # Handle both scalar and tensor cases
            for index, error in enumerate(group_error):
                num_targets = batch_target["num_targets"][index]
                if num_targets == 0:
                    continue  # Skip samples with no targets
                tgt_start = sum(batch_target["num_targets"][:index])
                tgt_end = tgt_start + num_targets
                error_list.append(
                    (
                        error / num_targets,
                        batch_sample[index],
                        batch_target_sorted[tgt_start:tgt_end],
                        batch_prediction_sorted[tgt_start:tgt_end],
                    )
                )

        # # For demonstration purposes, process only one batch
        # if batch_index >= 50:
        #     break

    # After processing all batches
    print(f"Number of unmatched predictions: {len(unmatched_predictions)}")
    print(f"Number of unmatched targets: {len(unmatched_targets)}")
    print(f"Number of samples with low cost: {len(error_list)}")

    # Sanity check: Ensure unmatched_predictions and unmatched_targets are equal
    assert len(unmatched_predictions) == len(unmatched_targets), (
        f"Total unmatched_predictions ({len(unmatched_predictions)}) does not match "
        f"unmatched_targets ({len(unmatched_targets)})"
    )

    # Initialize lists for predictions and targets
    parameter_predictions = []
    parameter_targets = []
    class_prediction_list = []
    class_target_list = []
    labels_dict = dict(cfg["mult_class_indices"])

    # Proceed only if error_list has entries
    if error_list:
        # Get data transformation
        transform = learner.dls[0].dataset.dataset.transformation

        # Reverse the error list to get the worst predictions
        error_list = error_list[::-1]

        # Set device to CPU for plotting and further processing
        processing_device = "cpu"

        for idx, entry in enumerate(error_list):
            if len(entry) != 4:
                print(f"Skipping entry at index {idx} due to unexpected length: {len(entry)}")
                continue  # Skip entries that don't have exactly 4 elements

            error, sample, target, prediction = entry

            # Move tensors to CPU for further processing
            sample = sample.to(processing_device)
            normed_prediction = prediction.to(processing_device)
            normed_target = target.to(processing_device)

            # Split classification and regression components
            num_classes = len(cfg.mult_class_indices)
            normed_param_predictions = normed_prediction[..., num_classes:]
            normed_param_targets = normed_target[:, 1:]

            # Ensure class_prediction contains probabilities
            class_prediction = normed_prediction[..., :num_classes].cpu().numpy()
            class_prediction = softmax(class_prediction, axis=1)  # Apply softmax
            class_target = normed_target[:, 0].cpu().numpy()

            class_prediction_list.append(class_prediction)
            class_target_list.append(class_target)

            # Determine multiplet list based on class predictions
            multiplet_list = get_class(class_prediction, threshold=0.0)

            # Unnormalize parameter predictions
            unnormed_predictions, unnormed_targets = unnorm(
                normed_param_predictions, normed_param_targets, transform
            )

            # Collect parameters into flat lists
            parameter_predictions.extend(unnormed_predictions)
            parameter_targets.extend(unnormed_targets)

            points_per_ppm = cfg.model_params.backbone.input_length / (
                cfg.plotting.max_ppm - cfg.plotting.min_ppm
            )
            # Plot parameter predictions for the first 10 samples
            if idx < 10:
                plot_prediction(
                    idx,
                    sample.cpu().numpy(),
                    unnormed_predictions,
                    unnormed_targets,
                    class_target,
                    cfg.reg_param_indices,
                    multiplet_list=multiplet_list,
                    points_per_ppm=points_per_ppm,
                    class_names=cfg.mult_class_indices,
                )

        print(f"Total number of parameter predictions: {len(parameter_predictions)}")
        print(f"Total number of parameter targets: {len(parameter_targets)}")

        # Sanity check: Ensure predictions and targets are of equal length
        assert len(parameter_predictions) == len(parameter_targets), (
            "Mismatch between parameter predictions and targets."
        )

        # Check chemical shift targets
        if parameter_targets:
            # Extract chemical shift targets
            chem_shift_idx = cfg.reg_param_indices["center_position_in_points"]
            chemical_shift_targets = np.array([tgt[chem_shift_idx] for tgt in parameter_targets])

            # Identify chemical shift targets that are less than 20
            invalid_indices = np.where(chemical_shift_targets < 20)[0]
            if invalid_indices.size > 0:
                print(f"Invalid chemical shift targets indices: {invalid_indices}")
                print(
                    f"Invalid chemical shift targets values: {chemical_shift_targets[invalid_indices]}"
                )

                # Remove invalid chemical shift targets from the lists
                parameter_predictions = [
                    pred
                    for idx, pred in enumerate(parameter_predictions)
                    if idx not in invalid_indices
                ]
                parameter_targets = [
                    tgt for idx, tgt in enumerate(parameter_targets) if idx not in invalid_indices
                ]
        else:
            print("No chemical shift targets to process.")
    else:
        print("Error list is empty. No samples to process.")

    # Proceed only if there are parameter predictions and targets
    if parameter_predictions and parameter_targets:
        # Convert lists to NumPy arrays
        parameter_predictions = np.array(parameter_predictions)
        parameter_targets = np.array(parameter_targets)

        # Extract specific parameters
        cc_param_index = cfg.reg_param_indices["coupling_constant_3_in_points"]
        cc_parameter_predictions = parameter_predictions[:, cc_param_index]
        cc_parameter_targets = parameter_targets[:, cc_param_index]

        # Filter out zero values
        non_zero_indices = cc_parameter_targets != 0.0
        cc_parameter_predictions = cc_parameter_predictions[non_zero_indices]
        cc_parameter_targets = cc_parameter_targets[non_zero_indices]

        # Initialize a list to store the results
        metrics_results = []

        for param_name, index in cfg["reg_param_indices"].items():
            # Determine the scale based on the parameter name
            scale = "hz"

            # Extract the proper name without '_in_points'
            pretty_param_name = param_name.replace("_in_points", "").replace("_", " ")

            # Replace specific parameter names with more appropriate terms
            if "center position" in pretty_param_name:
                pretty_param_name = pretty_param_name.replace("center position", "Chemical Shift")
            elif "bounding box range" in pretty_param_name:
                pretty_param_name = pretty_param_name.replace(
                    "bounding box range", "Signal Region Width"
                )
            elif "line width" in pretty_param_name:
                pretty_param_name = pretty_param_name.replace("line width", "Line Width")

            if "coupling" in pretty_param_name:
                if index == 3:
                    # For coupling constants, use the pre-processed combined data
                    if cc_parameter_predictions.size > 0:
                        plot_predictions_vs_truth(
                            predictions=cc_parameter_predictions,
                            targets=cc_parameter_targets,
                            param_name="Coupling Constant",
                            scale=scale,
                            npoints=cfg.model_params.backbone.input_length,
                            base_freq_in_MHz=cfg.plotting.base_freq_in_MHz,
                            max_ppm=cfg.plotting.max_ppm,
                            min_ppm=cfg.plotting.min_ppm,
                        )
                        # Calculate R2, MAE, and Median Error for coupling constants
                        r2 = r2_score(cc_parameter_targets, cc_parameter_predictions)
                        mae = mean_absolute_error(cc_parameter_targets, cc_parameter_predictions)
                        median_error = np.median(
                            np.abs(cc_parameter_targets - cc_parameter_predictions)
                        )

                        # Store the results for coupling constants
                        metrics_results.append(
                            {
                                "Parameter": "Coupling Constant",
                                "R2": r2,
                                "MAE": mae,
                                "Median Error": median_error,
                            }
                        )

                    else:
                        print("No coupling constants to plot.")

            else:
                # For other parameters, process as usual
                predictions = parameter_predictions[:, index]
                targets = parameter_targets[:, index]

                # Call the plot function
                plot_predictions_vs_truth(
                    predictions=predictions,
                    targets=targets,
                    param_name=pretty_param_name,
                    scale=scale,
                    npoints=cfg.model_params.backbone.input_length,
                    base_freq_in_MHz=cfg.plotting.base_freq_in_MHz,
                    max_ppm=cfg.plotting.max_ppm,
                    min_ppm=cfg.plotting.min_ppm,
                )

                # Calculate R2, MAE, and Median Error
                r2 = r2_score(targets, predictions)
                mae = mean_absolute_error(targets, predictions)
                median_error = np.median(np.abs(targets - predictions))

                # Store the results
                metrics_results.append(
                    {
                        "Parameter": pretty_param_name,
                        "R2": r2,
                        "MAE": mae,
                        "Median Error": median_error,
                    }
                )

        # Create a DataFrame from the results
        metrics_df = pd.DataFrame(metrics_results)

        # Sort the DataFrame by R2 score for better readability
        metrics_df = metrics_df.sort_values(by="R2", ascending=False)

        # Display the DataFrame
        print(metrics_df)

        # Flatten class_target_list and class_prediction_list
        if class_prediction_list and class_target_list:
            try:
                # Flatten class_target_list into a 1D array
                class_targets_flat = np.concatenate(class_target_list)

                # Flatten class_prediction_list
                class_predictions_flat = np.concatenate(class_prediction_list, axis=0)
            except Exception as e:
                print(f"Error during concatenation: {e}")
                class_targets_flat = np.array([])
                class_predictions_flat = np.array([])
        else:
            print("Class prediction list or class target list is empty.")

        if class_targets_flat.size == 0 or class_predictions_flat.size == 0:
            print("Class targets or predictions are empty after concatenation.")
        else:
            # Calculate accuracy
            class_prediction_argmax = np.argmax(class_predictions_flat, axis=1)
            accuracy = accuracy_score(class_targets_flat, class_prediction_argmax)
            print(f"Argmax Accuracy without 'no spin': {accuracy}")

            # Plot confusion matrix
            plot_confusion_matrix(
                class_targets_flat,
                class_prediction_argmax,
                labels_dict,
            )
    else:
        print("No parameter predictions and targets to process.")

    # Constants and Labels
    labels_dict["no spin"] = NO_SPIN_CLASS

    print(f"Shape of class_prediction_list: {[pred.shape for pred in class_prediction_list]}")
    print(f"Shape of class_target_list: {[tgt.shape for tgt in class_target_list]}")

    # Handle unmatched targets by assigning 'no spin' predictions
    unmatched_targets_np = np.array([t.cpu().numpy() for t in unmatched_targets])
    unmatched_targets_classes = unmatched_targets_np.astype(int)

    # Flatten predictions and targets from the matched data
    class_predictions_flat = np.concatenate(
        [np.argmax(pred, axis=1) for pred in class_prediction_list]
    )
    class_targets_flat = np.concatenate(class_target_list)

    # Compute best thresholds using matched data
    best_thresholds = precision_recall_curve_plot(
        class_prediction_list,
        class_target_list,
        labels_dict,
        threshold_increment=0.05,
        apply_sigmoid=False,  # Set to True if predictions are logits
    )

    unmatched_predictions_logits = [pred[: len(labels_dict) - 1] for pred in unmatched_predictions]
    never_matched_predictions_logits = [
        pred[: len(labels_dict) - 1] for pred in never_matched_predictions
    ]
    class_prediction_list.extend(unmatched_predictions_logits)
    class_prediction_list.append(never_matched_predictions_logits)

    # Adjust matched predictions based on thresholds
    adjusted_predictions = []
    for batch_predictions in class_prediction_list:
        for prediction in batch_predictions:
            predicted_class = np.argmax(prediction)
            threshold = best_thresholds[predicted_class]
            if prediction[predicted_class] < threshold:
                adjusted_predictions.append(NO_SPIN_CLASS)
            else:
                adjusted_predictions.append(predicted_class)

    # Include unmatched 'no spin' predictions for unmatched targets
    adjusted_predictions_with_unmatched = np.array(adjusted_predictions)

    print(
        f"unmatched predition classes: {adjusted_predictions_with_unmatched[-len(unmatched_predictions_logits) :]}"
    )

    total_number_of_predictions = num_queries * batch_size * len(learner.dls[0])
    # After combining matched and unmatched targets
    negative_fill_class = np.full(
        total_number_of_predictions - len(class_targets_flat), NO_SPIN_CLASS
    )
    print(f"Negative fill class shape: {negative_fill_class.shape}")
    final_targets = np.concatenate(
        [class_targets_flat, unmatched_targets_classes, negative_fill_class]
    )

    print(f"Final targets shape: {final_targets.shape}")
    final_targets = final_targets.astype(int)  # Ensure integer type
    print(f"Final targets shape: {final_targets.shape}")

    # creat negative fill for unmatched predictions but with
    print(f"adjusted_predictions_with_unmatched shape: {adjusted_predictions_with_unmatched.shape}")
    adjusted_predictions_with_unmatched = np.concatenate(
        [
            adjusted_predictions_with_unmatched,
            np.full(
                len(final_targets) - len(adjusted_predictions_with_unmatched),
                NO_SPIN_CLASS,
            ),
        ]
    )
    print(f"Adjusted predictions shape: {adjusted_predictions_with_unmatched.shape}")
    adjusted_predictions_with_unmatched = adjusted_predictions_with_unmatched.astype(int)

    print(f"Adjusted predictions shape: {adjusted_predictions_with_unmatched.shape}")

    print(f"Total number of predictions: {total_number_of_predictions}")

    print(f"final_targets shape: {len(final_targets)}")
    print(f"adjusted_predictions_with_unmatched shape: {len(adjusted_predictions_with_unmatched)}")

    # Calculate metrics
    accuracy = accuracy_score(final_targets, adjusted_predictions_with_unmatched)
    print(f"Overall Accuracy: {accuracy}")
    print(
        classification_report(
            final_targets,
            adjusted_predictions_with_unmatched,
            target_names=list(labels_dict.keys()),
        )
    )

    # Plot confusion matrix
    plot_confusion_matrix(final_targets, adjusted_predictions_with_unmatched, labels_dict)

    # Additional counts and validation
    num_unmatched_targets = len(unmatched_targets_classes)
    total_instances = len(final_targets)
    print(f"Number of unmatched targets: {num_unmatched_targets}")
    print(f"Total instances in confusion matrix: {total_instances}")

    # Extract class predictions and targets for AP calculation
    print(f"Class prediction list shape: {len(class_prediction_list)}")
    # # class prediction list at index 0
    # print(f"Class prediction list shape: {class_prediction_list[0]}")
    # # class prediction lsit at index 10000
    # print(f"Class prediction list shape: {class_prediction_list[10000]}")
    class_predictions = np.concatenate([sublist for sublist in class_prediction_list])
    class_targets = np.concatenate([np.array(sublist) for sublist in class_target_list])

    # Extract center positions and widths (ranges) for ground truths and predictions
    center_pos = parameter_predictions[:, cfg.reg_param_indices["center_position_in_points"]]
    bbox_range = parameter_predictions[:, cfg.reg_param_indices["bounding_box_range_in_points"]]

    # Apply softmax to convert logits to probabilities
    exp_logits = np.exp(class_predictions - np.max(class_predictions, axis=1, keepdims=True))
    confidence_scores = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    # Filter the confidence scores using the class targets
    confidence_scores = np.array(
        [
            confidence_scores[i, int(target)]
            for i, target in enumerate(class_targets)
            if target != NO_SPIN_CLASS
        ]
    )

    # Construct the intervals and confidence for predictions
    predicted_intervals_with_confidence = [
        [center, width, conf]
        for center, width, conf in zip(center_pos, bbox_range, confidence_scores)
    ]

    # Create the ground truth intervals without confidence scores
    ground_truth_intervals = [
        [center, width]
        for center, width in zip(
            parameter_targets[:, cfg.reg_param_indices["center_position_in_points"]],
            parameter_targets[:, cfg.reg_param_indices["bounding_box_range_in_points"]],
        )
    ]

    # Define the overlap thresholds to evaluate at
    overlap_thresholds = np.linspace(0.1, 0.9, 9)

    matched_gt_preds = [
        (gt[0], gt[1], pred[0], pred[1], pred[2])
        for gt, pred in zip(ground_truth_intervals, predicted_intervals_with_confidence)
    ]

    # Call the function with the combined list
    calculate_ap_for_1d_detections(matched_gt_preds, overlap_thresholds)


if __name__ == "__main__":
    main()
