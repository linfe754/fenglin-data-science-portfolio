import os
import pandas as pd
from config import INPUT_DIR, OUTPUT_DIR, DATASETS, MODELS, INPUT_PATTERNS
from metrics_runner import run_simple_eval, run_bayesian_eval, run_ours_eval


# Choose dataset abbreviation: "AZ", "imdb", or "tweet"
ds_abbr = "AZ"
# Choose model: "Gemma", "Llama", "Llama3b", "Mistral", or "Qwen"
model = "Gemma"
# Choose method: "noshot", "base", "bayesPE_notshot", "bayesPE_baseaddgroup", "oursnoshot", "oursbaseaddgroup"
method = "bayesPE_baseaddgroup"

# ==== AUTOMATIC CONFIG ====
ds_info = DATASETS[ds_abbr]
label_map = ds_info["label_map"]
classes = ds_info["classes"]
n_prompts = ds_info["n_prompts"]
n_classes = ds_info["n_classes"]
dataset_name = ds_info["name"]

def build_path(pattern, model, ds):
    return os.path.join(INPUT_DIR, pattern.format(model=model, ds=ds))

# ==== METHOD DISPATCH ====

if method == "noshot":
    input_file = build_path(INPUT_PATTERNS["noshot"], model, ds_abbr)
    row = run_simple_eval(input_file, model, label_map, classes, dataset_name, "noshot")
    print(pd.DataFrame([row]))

elif method == "base":
    input_file = build_path(INPUT_PATTERNS["base"], model, ds_abbr)
    row = run_simple_eval(input_file, model, label_map, classes, dataset_name, "base")
    print(pd.DataFrame([row]))

elif method == "bayesPE_notshot":
    eval_file = build_path(INPUT_PATTERNS["bayesPE_val"], model, ds_abbr)
    group_file = build_path(INPUT_PATTERNS["bayesPE_group"], model, ds_abbr)
    rows = run_bayesian_eval(eval_file, group_file, model, label_map, classes, dataset_name, "BayesPENotshot", group_col="group")
    print(pd.DataFrame(rows))

elif method == "bayesPE_baseaddgroup":
    eval_file = build_path(INPUT_PATTERNS["bayesPE_baseaddgroup_val"], model, ds_abbr)
    group_file = build_path(INPUT_PATTERNS["bayesPE_baseaddgroup_group"], model, ds_abbr)
    rows = run_bayesian_eval(eval_file, group_file, model, label_map, classes, dataset_name, "BayesPEonBaseAddGroup", group_col="group_id")
    print(pd.DataFrame(rows))

elif method == "oursnoshot":
    prior_file = build_path(INPUT_PATTERNS["oursnoshot_prior"], model, ds_abbr)
    group_file = build_path(INPUT_PATTERNS["oursnoshot_group"], model, ds_abbr)
    row = run_ours_eval(prior_file, group_file, model, label_map, classes, dataset_name, "oursnoshot", n_prompts, n_classes)
    print(pd.DataFrame([row]))

elif method == "oursbaseaddgroup":
    prior_file = build_path(INPUT_PATTERNS["oursbaseaddgroup_prior"], model, ds_abbr)
    group_file = build_path(INPUT_PATTERNS["oursbaseaddgroup_group"], model, ds_abbr)
    row = run_ours_eval(prior_file, group_file, model, label_map, classes, dataset_name, "oursbaseaddgroup", n_prompts, n_classes)
    print(pd.DataFrame([row]))

else:
    raise ValueError(f"Unknown method: {method}")
