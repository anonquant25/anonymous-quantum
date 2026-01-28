import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc

from safe.cramer import gini_via_lorenz, cvm1_concordance_weighted


def rga_cramer(y, yhat):
    """
    RGA using Cramér–von Mises (CvM) distance
    RGA = 1 - CvM(y, yhat) / G(y)

    Parameters
    ----------
    y : array-like
        True values
    yhat : array-like
        Predicted values

    Returns
    -------
    float
        RGA score
    """
    g = gini_via_lorenz(y)
    if not np.isfinite(g) or g == 0:
        return np.nan

    cvm = cvm1_concordance_weighted(y, yhat)
    if not np.isfinite(cvm):
        return np.nan

    return 1 - cvm / g


def partial_rga_cramer(y, yhat, n_segments):
    """
    Decompose RGA into partial contributions across segments.

    Parameters
    ----------
    y : array-like
        True values
    yhat : array-like
        Predicted values
    n_segments : int
        Number of segments to decompose into

    Returns
    -------
    dict
        Dictionary containing:
        - 'full_rga': RGA score
        - 'partial_rga': Partial RGA contributions for each segment
        - 'cumulative_vector': Cumulative vector [RGA, RGA-RGA_1, ..., 0]
        - 'segment_indices': List of index ranges for each segment
    """
    y = np.asarray(y, dtype=float).reshape(-1)
    yhat = np.asarray(yhat, dtype=float).reshape(-1)
    mask = ~np.isnan(y) & ~np.isnan(yhat)
    y = y[mask]
    yhat = yhat[mask]

    n = len(y)
    if n == 0:
        return {
            'full_rga': np.nan,
            'partial_rga': np.array([]),
            'cumulative_vector': np.array([]),
            'segment_indices': []
        }

    # Calculate full RGA
    full_rga = rga_cramer(y, yhat)
    full_gini = gini_via_lorenz(y)

    if not np.isfinite(full_rga) or not np.isfinite(full_gini) or full_gini == 0:
        return {
            'full_rga': full_rga,
            'partial_rga': np.array([np.nan] * n_segments),
            'cumulative_vector': np.array([np.nan] * (n_segments + 1)),
            'segment_indices': []
        }

    # Sort by predictions (descending)
    ord_yhat_desc = np.argsort(yhat)[::-1]
    y_sorted = y[ord_yhat_desc]
    yhat_sorted = yhat[ord_yhat_desc]

    # Divide into segments
    segment_size = n // n_segments
    remainder = n % n_segments

    partial_rga = []
    segment_indices = []

    start_idx = 0
    for k in range(n_segments):
        # Remainder across first segments
        current_size = segment_size + (1 if k < remainder else 0)
        end_idx = start_idx + current_size

        segment_indices.append((start_idx, end_idx))

        # Extract segment
        y_segment = y_sorted[start_idx:end_idx]
        yhat_segment = yhat_sorted[start_idx:end_idx]

        # Calculate RGA for this segment
        segment_rga = rga_cramer(y_segment, yhat_segment)

        # Weight by segment's contribution to total Gini
        segment_gini = gini_via_lorenz(y_segment)

        if np.isfinite(segment_gini) and segment_gini > 0:
            # Normalize by segment size relative to total
            weight = len(y_segment) / n
            weighted_contribution = segment_rga * segment_gini * weight / full_gini
        else:
            weighted_contribution = 0.0

        partial_rga.append(weighted_contribution)
        start_idx = end_idx

    partial_rga = np.array(partial_rga)

    # Normalize
    sum_partial = np.sum(partial_rga)
    if sum_partial > 0:
        partial_rga = partial_rga * (full_rga / sum_partial)

    # Build cumulative vector
    cumulative_vector = np.zeros(n_segments + 1)
    cumulative_vector[0] = full_rga

    cumsum = 0.0
    for k in range(n_segments):
        cumsum += partial_rga[k]
        cumulative_vector[k + 1] = full_rga - cumsum

    return {
        'full_rga': full_rga,
        'partial_rga': partial_rga,
        'cumulative_vector': cumulative_vector,
        'segment_indices': segment_indices
    }


def rga_cramer_multiclass(y_labels, prob_matrix, class_order=None, verbose=False):
    """
    Calculate RGA for multiclass classification using one-vs-rest approach.

    Parameters
    ----------
    y_labels : array-like
        True class labels
    prob_matrix : array-like, shape (n_samples, n_classes)
        Predicted probabilities for each class.
        Columns must correspond to `class_order` if provided,
        or to sorted unique classes in y_labels if not.
    class_order : array-like, optional
        Order of classes corresponding to prob_matrix columns (.classes_).
        If None, assumes prob_matrix columns match sorted unique(y_labels).
    verbose : bool, optional
        Print detailed information

    Returns
    -------
    tuple
        (rga_weighted, rga_per_class, class_weights, classes_used)
        - rga_weighted: Overall weighted RGA score
        - rga_per_class: RGA score for each class
        - class_weights: Weight of each class
        - classes_used: The class order used for computation
    """
    y_labels = np.asarray(y_labels)
    prob_matrix = np.asarray(prob_matrix)

    # Determine class order
    if class_order is None:
        if verbose:
            print('WARNING: class_order is not provided. Assuming prob_matrix columns match sorted unique classes.')
        class_order = np.unique(y_labels)
    else:
        class_order = np.asarray(class_order)

    n_classes = len(class_order)

    # Validate dimensions
    if prob_matrix.shape[1] != n_classes:
        raise ValueError(
            f'prob_matrix has {prob_matrix.shape[1]} columns but class_order has {n_classes} classes.'
        )

    rgas = []
    weights = []

    for k, c in enumerate(class_order):
        # One-vs-rest encoding
        y_bin = np.equal(y_labels, c).astype(np.float32)
        yhat_c = prob_matrix[:, k]

        if np.sum(y_bin) == 0:
            if verbose:
                print(f'Warning: Class {c} has zero samples. Skipping.')
            rgas.append(0.0)
            weights.append(0.0)
            continue

        rga_k = rga_cramer(y_bin, yhat_c)
        rgas.append(rga_k)
        weights.append(np.mean(y_bin))

    rgas = np.array(rgas)
    weights = np.array(weights)

    # Weighted average
    rga_weighted = np.nansum(rgas * weights) / np.nansum(weights)

    return rga_weighted, rgas, weights, class_order


def partial_rga_cramer_multiclass(y_labels, prob_matrix, n_segments, class_order=None, verbose=False):
    """
    Calculate partial RGA curves for multiclass classification.

    Parameters
    ----------
    y_labels : array-like
        True class labels
    prob_matrix : array-like, shape (n_samples, n_classes)
        Predicted probabilities for each class.
    n_segments : int
        Number of segments for partial decomposition
    class_order : array-like, optional
        Order of classes corresponding to prob_matrix columns.
    verbose : bool, optional
        Print detailed information

    Returns
    -------
    dict
        Dictionary containing:
        - 'cumulative_vector': Weighted average cumulative vector
        - 'per_class_vectors': Cumulative vectors for each class
        - 'class_weights': Weight of each class
        - 'classes': Class order used
    """
    y_labels = np.asarray(y_labels)
    prob_matrix = np.asarray(prob_matrix)

    # Determine class order
    if class_order is None:
        if verbose:
            print('WARNING: class_order is not provided. Assuming prob_matrix columns match sorted unique classes.')
        class_order = np.unique(y_labels)
    else:
        class_order = np.asarray(class_order)

    cum_vectors = []
    class_weights = []

    for k, c in enumerate(class_order):
        y_bin = np.equal(y_labels, c).astype(np.float32)
        yhat_c = prob_matrix[:, k]

        res = partial_rga_cramer(y_bin, yhat_c, n_segments)
        cum_vectors.append(res['cumulative_vector'])
        class_weights.append(np.mean(y_bin))

    cum_vectors = np.vstack(cum_vectors)
    class_weights = np.array(class_weights)

    # Weighted average across classes
    weighted_curve = np.average(cum_vectors, weights=class_weights, axis=0)

    return {
        'cumulative_vector': weighted_curve,
        'per_class_vectors': cum_vectors,
        'class_weights': class_weights,
        'classes': class_order
    }


# Evaluation Function
def evaluate_rga_multiclass(y_labels, prob_matrix, class_order=None, n_segments=10,
                            model_name='Model', plot=True, fig_size=(12, 5),
                            verbose=True, save_path=None):
    """
    RGA evaluation for multiclass classification.

    Parameters
    ----------
    y_labels : array-like
        True class labels
    prob_matrix : array-like, shape (n_samples, n_classes)
        Predicted probabilities for each class
    class_order : array-like, optional
        Order of classes corresponding to prob_matrix columns.
        For sklearn models, pass `model.classes_`.
        For PyTorch models, pass the class order used in output layer, like np.array([0, 1, 2, ...]).
    n_segments : int, optional
        Number of segments for partial RGA decomposition
    model_name : str, optional
        Name of the model for display
    plot : bool, optional
        Whether to generate visualization
    fig_size : tuple, optional
        Figure size for plots
    verbose : bool, optional
        Print detailed results
    save_path :
        Path for saving the plot

    Returns
    -------
    dict
        Comprehensive results dictionary containing:
        - 'rga_full': Overall RGA score
        - 'rga_per_class': RGA for each class
        - 'class_weights': Weight of each class
        - 'aurga': Area under RGA curve
        - 'cumulative_vector': Cumulative RGA vector
        - 'per_class_vectors': Per-class cumulative vectors
        - 'classes': Class order used
    """
    # Calculate full RGA
    rga_full, rga_per_class, class_weights, classes_used = rga_cramer_multiclass(
        y_labels, prob_matrix, class_order=class_order, verbose=verbose
    )

    # Calculate partial RGA
    partial_results = partial_rga_cramer_multiclass(
        y_labels, prob_matrix, n_segments, class_order=class_order, verbose=verbose
    )

    cumulative_vector = partial_results['cumulative_vector']
    x_axis = np.linspace(0, 1, len(cumulative_vector))

    # Calculate AURGA
    aurga = auc(x_axis, cumulative_vector)

    # Print results
    if verbose:
        print(f'RGA Evaluation: {model_name}')
        print(f'Full RGA: {rga_full:.4f}')
        print(f'AURGA: {aurga:.4f}')
        print(f'\nClass order: {classes_used}')
        print('\nPer-Class RGA:')
        for i, (cls, rga_val, weight) in enumerate(
                zip(classes_used, rga_per_class, class_weights)
        ):
            print(f'Class {cls}: RGA={rga_val:.4f}, Weight={weight:.4f}')

    # Visualization
    if plot:
        plt.figure(figsize=fig_size)
        plt.plot(x_axis, cumulative_vector, marker='o', linewidth=2.5, markersize=6, color='steelblue',
            label=f'{model_name} (AURGA={aurga:.3f})'
        )

        plt.fill_between(x_axis,0, cumulative_vector, alpha=0.2, color='steelblue')
        plt.xlabel('Fraction of Data Removed', fontsize=11, fontweight='bold')
        plt.ylabel('RGA Score', fontsize=11, fontweight='bold')
        plt.title('RGA Curve', fontsize=12, fontweight='bold')
        plt.grid(alpha=0.3, linestyle='--')
        plt.xlim([0, 1])
        max_val = np.nanmax(cumulative_vector)
        plt.ylim([0, max_val * 1.1 if np.isfinite(max_val) else 1])
        plt.legend(fontsize=10)
        plt.tight_layout()
        if save_path is None:
            plt.show()
        else:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.close()

    return {
        'rga_full': rga_full,
        'rga_per_class': rga_per_class,
        'class_weights': class_weights,
        'aurga': aurga,
        'cumulative_vector': cumulative_vector,
        'per_class_vectors': partial_results['per_class_vectors'],
        'classes': classes_used
    }


def compare_models_rga(models_dict, y_labels, n_segments=10,
                        fig_size=(14, 6), verbose=True, save_path=None):
    """
    Compare multiple models using RGA metrics.

    Parameters
    ----------
    models_dict : dict
        Dictionary mapping model names to tuples of (prob_matrix, class_order).
        Example: {
            'Random Forest': (rf.predict_proba(x_test), rf.classes_),
            'Neural Network': (nn_probs, np.array([0, 1, 2]))
        }
    y_labels : array-like
        True class labels
    n_segments : int, optional
        Number of segments for partial RGA
    fig_size : tuple, optional
        Figure size for comparison plot
    verbose : bool, optional
        Print detailed comparison
    save_path :
        Path for saving the plot

    Returns
    -------
    dict
        Comparison results for all models
    """
    results = {}

    for model_name, (prob_matrix, class_order) in models_dict.items():
        if verbose:
            print(f'\nEvaluating {model_name}...')

        result = evaluate_rga_multiclass(
            y_labels, prob_matrix, class_order=class_order,
            n_segments=n_segments, model_name=model_name,
            plot=False, verbose=verbose, save_path=save_path
        )
        results[model_name] = result

    # Comparison plot
    plt.figure(figsize=fig_size)

    cmap = plt.get_cmap('tab10')
    colors = cmap(np.linspace(0, 1, len(results)))

    for (model_name, result), color in zip(results.items(), colors):
        x_axis = np.linspace(0, 1, len(result['cumulative_vector']))
        plt.plot(x_axis, result['cumulative_vector'], "-o", linewidth=2.5, markersize=5,
            color=color, label=f"{model_name} (AURGA={result['aurga']:.3f})"
        )
    plt.xlabel('Fraction of Data Removed', fontsize=11, fontweight='bold')
    plt.ylabel('RGA Score', fontsize=11, fontweight='bold')
    plt.title('RGA Curves Comparison', fontsize=12, fontweight='bold')
    plt.grid(alpha=0.3, linestyle="--")
    plt.xlim([0, 1])
    plt.legend(fontsize=9)
    plt.tight_layout()
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.close()

    if verbose:
        model_names = list(results.keys())
        aurga_scores = [results[n]['aurga'] for n in model_names]
        rga_scores = [results[n]['rga_full'] for n in model_names]

        print('RGA Comparison Summary')
        for n, w, a in zip(model_names, rga_scores, aurga_scores):
            print(f'{n}: RGA={w:.4f}, AURGA={a:.4f}')

    return results