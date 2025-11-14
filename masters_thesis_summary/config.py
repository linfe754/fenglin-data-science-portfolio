import os

CWD = os.getcwd()
INPUT_DIR = os.path.join(CWD, "inputFiles")
OUTPUT_DIR = os.path.join(CWD, "results")

MODELS = ["Gemma", "Llama", "Llama3b", "Mistral", "Qwen"]

DATASETS = {
    "AZ": {
        "name": "AmazonReviews",
        "label_map": {"positive": 0, "negative": 1},
        "classes": ["Positive", "Negative"],
        "n_prompts": 10,
        "n_classes": 2
    },
    "imdb": {
        "name": "IMDBReviews",
        "label_map": {"positive": 0, "negative": 1},
        "classes": ["Positive", "Negative"],
        "n_prompts": 10,
        "n_classes": 2
    },
    "tweet": {
        "name": "TweetEval",
        "label_map": {"Positive": 0, "Negative": 1, "Neutral": 2},
        "classes": ["Positive", "Negative", "Neutral"],
        "n_prompts": 10,
        "n_classes": 3
    }
}

OUTPUT_PATTERNS = {
    # noshot/base
    "noshot": "{ds}_all_noshot_metrics.csv",
    "base": "{ds}_all_base_metrics.csv",
    # bayesian
    "bayesPENotshot": "{ds}_all_bayesPENotshot_metrics.csv",
    "bayesPE_baseaddgroup": "{ds}_all_bayesPE_baseaddgroup_metrics.csv",
    # ours
    "oursnoshot": "{ds}_all_oursnoshot_metrics.csv",
    "oursbaseaddgroup": "{ds}_all_oursbaseaddgroup_metrics.csv"
}

# File patterns for input files
INPUT_PATTERNS = {
    "noshot": "resultsNoshot{model}_{ds}.csv",
    "base": "resultsBase{model}_{ds}.csv",
    "bayesPE_val": "resultsGroups{model}BPE_{ds}.csv",
    "bayesPE_group": "resultsGroups{model}_{ds}.csv",
    "bayesPE_baseaddgroup_val": "resultsBaseAddGroup{model}BPE_{ds}.csv",
    "bayesPE_baseaddgroup_group": "resultsBaseAddGroup{model}_{ds}.csv",
    "oursnoshot_prior": "resultsNoshot{model}_{ds}.csv",
    "oursnoshot_group": "resultsGroups{model}_{ds}.csv",
    "oursbaseaddgroup_prior": "resultsBase{model}_{ds}.csv",
    "oursbaseaddgroup_group": "resultsBaseAddGroup{model}_{ds}.csv"
}
