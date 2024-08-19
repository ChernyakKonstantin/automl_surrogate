# Datasets root structure:
```
test_<DATASET_NAME>_fold<FOLD_I>.npy
testy_<DATASET_NAME>_fold<FOLD_I>.npy
train_<DATASET_NAME>_fold<FOLD_I>.npy
trainy_<DATASET_NAME>_fold<FOLD_I>.npy
...
```

# Methods evaluations

In all evaluation scripts one should specify `--datasets_root <PATH>` and `--logfile <PATH>` arguments.

In all parsing scripts one should specify `--input_logfile <PATH>` and `--output_csv_file`.


## Default LGBM (Default baseline) evaluation

1. Run `lgbm/evaluate.py` to obtain a log-file.
2. Run `lgbm/parsers/parse_lgbm_evaluation.py` to obtain a csv-file.

## Best LGBM (Best baseline) evaluation

1. Run `lgbm/evaluate_best_configurations.py` to obtain a log-file. We provide selected best configurations in `lgbm/best_lgbm_configurations.csv`
2. Run `lgbm/parsers/parse_best_lgbm_evaluation.py` to obtain a csv-file.

## Best Pipeline evaluation

1. Run `<anon_framework>/evaluate.py --predefined_models_root <PATH>` to obtain a log-file. We provide selected best configurations in `<anon_framework>/best_pipelines.zip` (file should be unzipped).
2. Run `<anon_framework>/parsers/parse_best_pipeline_evaluation.py` to obtain a csv-file.

## <Anon_framework> w/o surrogate evaluation

1. Run `<anon_framework>/evaluate.py` to obtain a log-file.
2. Run `<anon_framework>/parsers/parse_<anon_framework>_evaluation.py` to obtain a csv-file.

## <Anon_framework> w/ surrogate evaluation

1. Run `<anon_framework>/evaluate.py --surrogate_config_file <PATH>` to obtain a log-file.
2. Run `<anon_framework>/parsers/parse_<anon_framework>_evaluation.py` to obtain a csv-file.

## Autogluon evaluation

1. Run `autogluon/evaluate.py` to obtain a log-file.
2. Run `autogluon/parsers/parse_autogluon_evaluation.py` to obtain a csv-file.