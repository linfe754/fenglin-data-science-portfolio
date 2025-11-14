import os
import pandas as pd
from config import INPUT_DIR, OUTPUT_DIR, MODELS, DATASETS, OUTPUT_PATTERNS, INPUT_PATTERNS
from metrics_runner import run_simple_eval, run_bayesian_eval, run_ours_eval

def build_path(pattern, model, ds):
    return os.path.join(INPUT_DIR, pattern.format(model=model, ds=ds))

def build_output(pattern, ds):
    return os.path.join(OUTPUT_DIR, pattern.format(ds=ds))

def main():
    for ds_abbr, ds_info in DATASETS.items():
        name = ds_info["name"]
        label_map = ds_info["label_map"]
        classes = ds_info["classes"]
        n_prompts = ds_info["n_prompts"]
        n_classes = ds_info["n_classes"]

        # --- NOSHOT ---
        results = []
        for model in MODELS:
            f = build_path(INPUT_PATTERNS["noshot"], model, ds_abbr)
            row = run_simple_eval(f, model, label_map, classes, name, "noshot")
            results.append(row)
        pd.DataFrame(results).to_csv(build_output(OUTPUT_PATTERNS["noshot"], ds_abbr), index=False)

        # --- BASE ---
        results = []
        for model in MODELS:
            f = build_path(INPUT_PATTERNS["base"], model, ds_abbr)
            row = run_simple_eval(f, model, label_map, classes, name, "base")
            results.append(row)
        pd.DataFrame(results).to_csv(build_output(OUTPUT_PATTERNS["base"], ds_abbr), index=False)

        # --- BAYESIAN (Notshot/Group) ---
        all_rows = []
        for model in MODELS:
            eval_file = build_path(INPUT_PATTERNS["bayesPE_val"], model, ds_abbr)
            group_file = build_path(INPUT_PATTERNS["bayesPE_group"], model, ds_abbr)
            rows = run_bayesian_eval(
                eval_file, group_file, model,
                label_map, classes, name,
                "BayesPENotshot", group_col="group"
            )
            all_rows.extend(rows)   # <--- flatten!
        pd.DataFrame(all_rows).to_csv(build_output(OUTPUT_PATTERNS["bayesPENotshot"], ds_abbr), index=False)

        # --- BAYESIAN (BaseAddGroup) ---
        all_rows = []
        for model in MODELS:
            eval_file = build_path(INPUT_PATTERNS["bayesPE_baseaddgroup_val"], model, ds_abbr)
            group_file = build_path(INPUT_PATTERNS["bayesPE_baseaddgroup_group"], model, ds_abbr)
            rows = run_bayesian_eval(
                eval_file, group_file, model,
                label_map, classes, name,
                "BayesPEonBaseAddGroup", group_col="group_id"
            )
            all_rows.extend(rows)   # <--- flatten!
        pd.DataFrame(all_rows).to_csv(build_output(OUTPUT_PATTERNS["bayesPE_baseaddgroup"], ds_abbr), index=False)

        # --- OURS (noshot) ---
        results = []
        for model in MODELS:
            prior_file = build_path(INPUT_PATTERNS["oursnoshot_prior"], model, ds_abbr)
            group_file = build_path(INPUT_PATTERNS["oursnoshot_group"], model, ds_abbr)
            row = run_ours_eval(prior_file, group_file, model, label_map, classes, name, "oursnoshot", n_prompts, n_classes)
            results.append(row)
        pd.DataFrame(results).to_csv(build_output(OUTPUT_PATTERNS["oursnoshot"], ds_abbr), index=False)

        # --- OURS (baseaddgroup) ---
        results = []
        for model in MODELS:
            prior_file = build_path(INPUT_PATTERNS["oursbaseaddgroup_prior"], model, ds_abbr)
            group_file = build_path(INPUT_PATTERNS["oursbaseaddgroup_group"], model, ds_abbr)
            row = run_ours_eval(prior_file, group_file, model, label_map, classes, name, "oursbaseaddgroup", n_prompts, n_classes)
            results.append(row)
        pd.DataFrame(results).to_csv(build_output(OUTPUT_PATTERNS["oursbaseaddgroup"], ds_abbr), index=False)

if __name__ == "__main__":
    main()
