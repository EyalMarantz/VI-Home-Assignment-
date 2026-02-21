import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from features import build_features
from train_evaluate import train, cross_validate, evaluate_uplift, predict_uplift, model_selection, choose_n
import json
from tqdm import trange, tqdm



def plot_results():
    results = {"xgboost": 0.07, "placebo": 0.02, "random": 0.01}





def pipeline(param_path="xgb_base_features") -> None:
    """
    Generic pipeline to run the entire process: build features, train models and evaluate model.

    :param param_path: path to the JSON file containing the best parameters for treatment and control models. The JSON file should have the following structure:
    :return: None
    """
    train_data = build_features_reasonable(dataset="train")
    json_path = os.path.join("experiments", f"{param_path}_best_params.json")

    with open(json_path, "r") as f:
        best_params = json.load(f)
        treatment_model = XGBClassifier(**best_params["treatment"])
        control_model = XGBClassifier(**best_params["control"])


    #test_data, _ = build_features(dataset="test", normelize=normelize, scaler=fited_scaler)
    cv_score = cross_validate(train_data, n_splits=5, model=XGBClassifier,
                              t_param=best_params["treatment"], c_param=best_params["control"])
    print(f"Average Uplift at top 20% across folds (CV): {cv_score:.4f}")





def run_features_experiments():
    """
    Run experiments to compare different feature sets and models.
    This function will build features for each combination of model and feature set, train the model using
    cross-validation, and save the results in a CSV file.
    """
    feature_sets = ["base", "base_content", "base_content_intersection"]
    model = ["xgb"]
    res = []
    for model in model:
        for fs in feature_sets:
            train_df = build_features_reasonable(dataset="train", feature_set=fs)
            experiment_name = f"{model}_{fs}_features"
            print(f"Running baseline for experiment: {experiment_name} with feature set: {fs}")
            cv_score = model_selection(train_df, num_iters=100, experiment_name=experiment_name, base_line=model == "logistic")
            res.append({"model": model, "feature_set": fs, "experiment_name": experiment_name, "cv_score": cv_score})

    res = pd.DataFrame(res)
    pd.DataFrame.to_csv(res, "experiment_results.csv", index=False)

def run_placebo_experoment(param_path, num_repeats=50) -> None:
    """
    Run placebo experiment to evaluate the robustness of the model.
    :param param_path: path to the JSON file containing the best parameters for treatment and control models. The JSON file should have the following structure:
    :param num_repeats: number of times to repeat the placebo experiment to get a distribution of scores. Default is 50.
    :return:
    """
    train_df = build_features(dataset="train", feature_set="base")
    json_path = os.path.join("experiments", f"{param_path}_best_params.json")

    with open(json_path, "r") as f:
        best_params = json.load(f)
        treatment_model = XGBClassifier(**best_params["treatment"])
        control_model = XGBClassifier(**best_params["control"])

    placebo_scores = []
    random_scores = []
    for i in trange(num_repeats):
        cv_score = cross_validate(train_df,n_splits=5, model=XGBClassifier,
                                  t_param=best_params["treatment"],
                                  c_param=best_params["control"], placebo=True)
        placebo_scores.append(cv_score)

    print(f"Placebo experiment results for {param_path}: Mean Uplift at top 20% across folds: {np.mean(placebo_scores):.4f}, Std: {np.std(placebo_scores):.4f}")

    res = pd.DataFrame({"placebo_score": placebo_scores})
    pd.DataFrame.to_csv(res, f"placebo_experiment_results.csv", index=False)


if __name__ == '__main__':
    # run_features_experiments()
    run_placebo_experoment("xgb_base_content_features")
    #extra_featurestrain_df = build_features_reasonable(dataset="train", feature_set="base_content")
    #run_eda()
    #pipeline(param_path="xgb_base_content_features")
    #choose_n(train_df, param_path="xgb_base_content_features", run=False)

