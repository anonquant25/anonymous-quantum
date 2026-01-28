import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import auc

from safe.cramer import gini_via_lorenz, cvm1_concordance_weighted
from safe.utils import apply_patch_occlusion, get_predictions_from_features, apply_importance_masking


def rge_cramer(pred, pred_reduced):
    """
    RGE which compares original predictions with perturbed predictions.

    Parameters
    ----------
    pred : array-like
        Predictions from full model
    pred_reduced : array-like
        Predictions from reduced model

    Returns
    -------
    float
        RGE score
    """
    g = gini_via_lorenz(pred)
    if not np.isfinite(g) or g == 0:
        return np.nan
    cvm = cvm1_concordance_weighted(pred, pred_reduced)
    if not np.isfinite(cvm):
        return np.nan
    return cvm / g


def rge_cramer_multiclass(prob_full, prob_reduced, class_weights=None, verbose=False):
    """
    Calculate RGE for multiclass classification.
    Measures impact of feature removal/occlusion on predictions.
    Use align_proba_to_class_order() before calling this function.

    Parameters
    ----------
    prob_full : array-like, shape (n_samples, n_classes)
        Predictions from original model
    prob_reduced : array-like, shape (n_samples, n_classes)
        Predictions from occluded model
    class_weights : array-like, optional
        Custom weights for each class. If None, uses uniform weighting.
    verbose : bool, optional
        Print detailed information

    Returns
    -------
    tuple
        (rge_weighted, rge_per_class, weights_used)
        - rge_weighted: Overall weighted RGE score
        - rge_per_class: RGE score for each class
        - weights_used: Weights used for each class
    """
    prob_full = np.asarray(prob_full)
    prob_reduced = np.asarray(prob_reduced)

    n_samples, n_classes = prob_full.shape

    if prob_reduced.shape != prob_full.shape:
        raise ValueError(
            f'Shape mismatch: prob_full {prob_full.shape} and prob_reduced {prob_reduced.shape}'
        )

    # Set up class weights
    if class_weights is None:
        class_weights = np.ones(n_classes) / n_classes
    else:
        class_weights = np.asarray(class_weights)
        if len(class_weights) != n_classes:
            raise ValueError(
                f'class_weights length {len(class_weights)} does not match n_classes {n_classes}'
            )

    rges = []

    for k in range(n_classes):
        pred_full = prob_full[:, k]
        pred_reduced = prob_reduced[:, k]

        # RGE uses same computation as RGR
        rge_k = 1 - rge_cramer(pred_full, pred_reduced)
        rges.append(rge_k)

        if verbose:
            print(f'Class {k}: RGE = {rge_k:.4f}')

    rges = np.array(rges)

    # Weighted average
    rge_weighted = np.nansum(rges * class_weights) / np.nansum(class_weights)

    return rge_weighted, rges, class_weights


def evaluate_rge_multiclass_occlusion(
        model, preprocess_fn, images_dataset, removal_fractions,
        model_class_order, class_order,
        model_type='sklearn', device=None,
        patch_size=32, batch_size=64,
        class_weights=None, model_name='Model', rga_full=None,
        occlusion_method='random', patch_rankings=None, patch_meta=None,
        plot=True, fig_size=(10, 6), verbose=True,
        random_seed=None, mask_value=0.0, save_path=None
):
    """
    Evaluate RGE across increasing occlusion levels and compute AURGE.

    Parameters
    ----------
    model :
        The classifier to evaluate (sklearn or torch)
    preprocess_fn :
        Callable mapping images tensor (N,C,H,W) -> features ndarray (N,D) ready for `model`.
        This is where you typically call: feature extractor -> PCA -> scaler.
    images_dataset :
        Torch dataset yielding images and possibly labels
    removal_fractions :
        Fractions of image area to occlude in [0,1]
    model_class_order :
        Model's class order (e.g. sklearn model.classes_)
    class_order :
        Canonical class order
    model_type :
        'sklearn' or 'pytorch'
    device :
        Torch device
    patch_size :
        Patch size for random occlusion or importance patching
    batch_size :
        Batch size for loading dataset and for prediction
    class_weights :
        Optional weights for RGE aggregation
    model_name :
        Name used for logging and plots
    rga_full :
        If provided, RGE curve is rescaled by this value (required by SAFE)
    occlusion_method :
        'random' or 'gradcam_most'
    patch_rankings, patch_meta :
        Required when occlusion_method is gradcam_
    plot :
        Whether to plot the RGE curve
    fig_size :
        Figure size for plotting
    verbose :
        Verbose logging
    random_seed :
        Seed used for random occlusion
    mask_value :
        Fill value for masked pixels when using constant baseline
    save_path :
        Path for saving the plot

    Returns
    -------
    dict
        Contains raw/rescaled RGE values, AURGE, per-class RGE, and metadata.
    """
    removal_fractions = np.asarray(removal_fractions, dtype=float)

    if occlusion_method in ('gradcam_most', 'gradcam_least'):
        if patch_rankings is None or patch_meta is None:
            raise ValueError('For Grad-CAM masking you must pass patch_rankings and patch_meta')

    if verbose:
        print(f'RGE Evaluation: {model_name}')
        print(f'Occlusion: {occlusion_method}')
        print(f'Testing {len(removal_fractions)} removal fractions')

    # Load all images once
    loader = DataLoader(images_dataset, batch_size=batch_size, shuffle=False)
    images_all = []
    for batch in loader:
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        images_all.append(x)
    images_all = torch.cat(images_all, dim=0)

    _, _, h, w = images_all.shape
    total_pixels = h * w
    patch_pixels = patch_size * patch_size

    # Baseline predictions
    if verbose:
        print('Extracting features from original images...')
    feat_full = preprocess_fn(images_all)
    prob_full = get_predictions_from_features(
        feat_full, model, model_class_order, class_order,
        model_type=model_type, device=device, batch_size=batch_size
    )

    rge_scores = []
    per_class_rge_list = []

    for frac in removal_fractions:
        if verbose:
            print(f'\nOcclusion level: {frac * 100:.0f}%')

        if occlusion_method == 'random':
            pixels_to_remove = int(frac * total_pixels)
            num_patches = pixels_to_remove // patch_pixels
            images_occ = apply_patch_occlusion(
                images_all, num_patches, patch_size,
                random_seed=random_seed, mask_value=mask_value
            )

        elif occlusion_method == 'gradcam_most':
            images_occ = apply_importance_masking(
                images_all, patch_rankings, patch_meta, frac,
                mask_strategy='most_important', mask_value=mask_value
            )

        else:
            raise ValueError(f'Unknown occlusion_method: {occlusion_method}')

        feat_occ = preprocess_fn(images_occ)
        prob_occ = get_predictions_from_features(
            feat_occ, model, model_class_order, class_order,
            model_type=model_type, device=device, batch_size=batch_size
        )

        rge_val, rge_per_class, _ = rge_cramer_multiclass(prob_full, prob_occ, class_weights=class_weights)
        rge_val = 0.0 if np.isnan(rge_val) else float(rge_val)

        rge_scores.append(rge_val)
        per_class_rge_list.append(rge_per_class)

        if verbose:
            print(f'RGE = {rge_val:.4f}')

    rge_scores = np.asarray(rge_scores, dtype=float)
    per_class_rge_list = np.asarray(per_class_rge_list)

    # Rescale by RGA
    rge_rescaled = rge_scores * float(rga_full) if (
                rga_full is not None and np.isfinite(rga_full)) else rge_scores

    # AUC on normalized x-axis
    max_frac = float(np.max(removal_fractions)) if len(removal_fractions) else 1.0
    x = removal_fractions / max_frac if max_frac > 0 else removal_fractions
    aurge = auc(x, rge_rescaled)

    if verbose:
        print(f'AURGE: {aurge:.4f}')

    if plot:
        plt.figure(figsize=fig_size)
        plt.plot(removal_fractions * 100, rge_rescaled, '-o', linewidth=2.5, markersize=6)
        plt.fill_between(removal_fractions * 100, 0, rge_rescaled, alpha=0.2)
        plt.xlabel('Occluded Image Area (%)', fontsize=11, fontweight='bold')
        plt.ylabel('RGE Score', fontsize=11, fontweight='bold')
        plt.title(f'RGE Curve: {model_name} ({occlusion_method})', fontsize=12, fontweight='bold')
        plt.grid(alpha=0.3, linestyle='--')
        plt.tight_layout()
        if save_path is None:
            plt.show()
        else:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.close()

    return {
        'rge_scores': rge_scores,
        'rge_rescaled': rge_rescaled,
        'aurge': aurge,
        'removal_fractions': removal_fractions,
        'per_class_rge': per_class_rge_list,
        'class_order': class_order,
        'occlusion_method': occlusion_method,
    }


def compare_models_rge(
        models_dict, images_dataset, removal_fractions, class_order,
        occlusion_method='random',
        patch_size=32, batch_size=64, class_weights=None,
        rga_dict=None, device=None, fig_size=(12, 6), verbose=True,
        random_seed=None, patch_rankings=None, patch_meta=None, save_path=None
):
    """
    Evaluate and plot RGE curves for multiple models.

    Parameters
    ----------
    models_dict :
        Mapping model_name -> (model, preprocess_fn, model_class_order, model_type)
    images_dataset :
        Dataset images
    removal_fractions :
        Occlusion fractions in [0,1]
    class_order :
        Canonical class order.
    occlusion_method :
        Single method for all models OR per-model dict
    patch_size :
        Patch size for random occlusion or importance patching
    batch_size :
        Batch size for loading dataset and for prediction
    class_weights :
        Optional weights for RGE aggregation
    rga_dict :
        Needed for rescaling
    device :
        Torch device
    fig_size :
        Figure size for plotting
    verbose :
        Verbose logging
    random_seed :
        Seed used for random occlusion
    patch_rankings, patch_meta :
        Shared Grad-CAM patch ranking info (compute once) for gradcam_ methods
    save_path :
        Path for saving the plot

    Returns
    -------
    dict
        Results per model name
    """
    if isinstance(occlusion_method, str):
        methods = {name: occlusion_method for name in models_dict}
    elif isinstance(occlusion_method, dict):
        methods = occlusion_method
    else:
        raise TypeError(
            'occlusion_method must be a string (single method) or a dict {model_name: method}.'
        )

    results = {}

    for name, (model, preprocess_fn, model_class_order, model_type) in models_dict.items():
        if verbose:
            print(f'\nEvaluating {name}')

        res = evaluate_rge_multiclass_occlusion(
            model=model,
            preprocess_fn=preprocess_fn,
            images_dataset=images_dataset,
            removal_fractions=removal_fractions,
            model_class_order=model_class_order,
            class_order=class_order,
            model_type=model_type,
            device=device,
            patch_size=patch_size,
            batch_size=batch_size,
            class_weights=class_weights,
            model_name=name,
            rga_full=(rga_dict.get(name) if rga_dict else None),
            occlusion_method=methods.get(name, 'random'),
            patch_rankings=patch_rankings,
            patch_meta=patch_meta,
            plot=False,
            verbose=verbose,
            random_seed=random_seed,
            save_path=save_path
        )
        results[name] = res

    # Plot comparison.
    plt.figure(figsize=fig_size)
    cmap = plt.get_cmap('tab10')
    colors = cmap(np.linspace(0, 1, len(results)))

    for (name, res), col in zip(results.items(), colors):
        plt.plot(
            res['removal_fractions'] * 100,
            res['rge_rescaled'],
            '-o',
            linewidth=2.2,
            markersize=5,
            color=col,
            label=f"{name} [{res['occlusion_method']}] (AURGE={res['aurge']:.3f})",
        )

    plt.xlabel('Occluded Image Area (%)', fontsize=11, fontweight='bold')
    plt.ylabel('RGE Score', fontsize=11, fontweight='bold')
    plt.title('RGE Curves Comparison', fontsize=12, fontweight='bold')
    plt.grid(alpha=0.3, linestyle="--")
    plt.legend(fontsize=9)
    plt.tight_layout()
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.close()

    if verbose:
        print('\nExplainability Comparison Summary (AURGE)')
        for name in results:
            print(f"{name:15s}: AURGE={results[name]['aurge']:.4f}")

    return results