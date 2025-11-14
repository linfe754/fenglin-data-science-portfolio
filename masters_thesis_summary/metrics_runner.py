import pandas as pd
import numpy as np
from evaluation import evaluate_all
from Bayes import optimise_weights
from scipy.stats import entropy
from scipy.special import softmax

def run_simple_eval(
    input_file,
    model,
    label_map,
    classes,
    dataset_name,
    eval_type
):
    """
    Evaluate noshot, base
    Returns a single dict row.
    """
    df = pd.read_csv(input_file)
    y_true = df["actual"].map(label_map).values
    p_hat = df[classes].values
    metrics = evaluate_all(y_true, p_hat)
    row = {"model": model, "dataset": dataset_name, "type": eval_type}
    row.update(metrics)
    return row

def run_bayesian_eval(
    eval_file,
    group_file,
    model,
    label_map,
    classes,
    dataset_name,
    eval_type,
    group_col="group",
    include_group_extremes=True
):
    """
    Evaluate Bayesian prompt ensemble and best group worst group
    Returns a list of dict rows: [ensemble, BestGroup, WorstGroup].
    """
    eval_data = pd.read_csv(eval_file)
    group_data = pd.read_csv(group_file)

    # --- BayesPE ---
    mats = [eval_data.pivot(index="sample_id", columns=group_col, values=cls).sort_index(axis=1).values for cls in classes]
    probs = np.stack(mats, axis=1)
    eps = 1e-12
    probs = np.clip(probs, eps, 1.0)
    gt_labels_val = eval_data.drop_duplicates("sample_id")["actual"].map(label_map).values
    weights = optimise_weights(probs, gt_labels_val)

    mats_eval = [group_data.pivot(index="sample_id", columns=group_col, values=cls).sort_index(axis=1).values for cls in classes]
    w = weights.reshape(1, -1)
    preds = [np.sum(m * w, axis=1) for m in mats_eval]
    y_true = group_data.drop_duplicates("sample_id")["actual"].map(label_map).values
    p_hat = np.stack(preds, axis=1)
    p_hat = p_hat / p_hat.sum(axis=1, keepdims=True)
    metrics_bayespe = evaluate_all(y_true, p_hat)
    row_main = {"model": model, "dataset": dataset_name, "type": eval_type}
    row_main.update(metrics_bayespe)

    rows = [row_main]

    # --- Best/Worst Group ---
    if include_group_extremes:
        y_val = eval_data.drop_duplicates("sample_id")["actual"].map(label_map).values
        group_scores = {}
        for group in eval_data[group_col].unique():
            df_group = eval_data[eval_data[group_col] == group].copy()
            y_pred = df_group[classes].values
            y_pred = y_pred / y_pred.sum(axis=1, keepdims=True)
            metrics = evaluate_all(y_val, y_pred)
            group_scores[group] = metrics

        # Find best and worst group by NLL
        best_group = min(group_scores, key=lambda g: group_scores[g]["NLL"])
        worst_group = max(group_scores, key=lambda g: group_scores[g]["NLL"])

        for group, type_name in zip([best_group, worst_group], ["BestGroup", "WorstGroup"]):
            group_df = group_data[group_data[group_col] == group].copy()
            p_hat = group_df[classes].values
            p_hat = p_hat / p_hat.sum(axis=1, keepdims=True)
            metrics = evaluate_all(y_true, p_hat)
            row = {"model": model, "dataset": dataset_name, "type": type_name}
            row.update(metrics)
            rows.append(row)

    return rows

def run_ours_eval(
    prior_file,
    group_file,
    model,
    label_map,
    classes,
    dataset_name,
    eval_type,
    n_prompts,
    n_classes
):
    """
    Evaluate using Entropy-weighted aggregation.
    Returns a single dict row.
    """
    prior_rows = pd.read_csv(prior_file)
    rows = pd.read_csv(group_file)

    posterior_probs = rows[classes].to_numpy().reshape([-1, n_prompts, n_classes])
    entropies = entropy(posterior_probs, axis=-1)
    prior_probs = prior_rows[classes].to_numpy()
    prior_entropy = entropy(prior_probs, axis=-1)
    eps = 1e-12
    log_prior = np.log(np.clip(prior_probs, eps, 1.0))
    log_posterior = np.log(np.clip(posterior_probs, eps, 1.0))
    beta = np.expand_dims(prior_entropy, axis=1) - entropies
    beta = np.clip(beta, 0, None)
    beta = beta / (beta.sum(axis=1, keepdims=True) + eps)
    likelihood = log_posterior - np.expand_dims(log_prior, axis=1)
    weighted_likelihood = np.expand_dims(beta, axis=-1) * likelihood
    likelihood_aggregated = np.sum(weighted_likelihood, axis=1) + log_prior
    inferred_probs = softmax(likelihood_aggregated, axis=-1)
    y_eval = prior_rows["actual"].map(label_map).values
    metrics = evaluate_all(y_eval, inferred_probs)
    row = {"model": model, "dataset": dataset_name, "type": eval_type}
    row.update(metrics)
    return row
