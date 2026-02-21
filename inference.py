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


def inference(test_data: pd.DataFrame, model_treat, model_control) -> pd.DataFrame:
    """
    Inference function to predict uplift scores for the test set using the trained treatment and control models.
    :param test_data: test data with features built using the same process as the training data.
    :param model_treat: model trained on the treatment group (outreach = 1) to predict the probability of churn.
    :param model_control: model trained on the control group (outreach = 0) to predict the probability of churn.
    :return: DataFrame with member_id and uplift_score, sorted by uplift_score in descending order.
    """
    X_test = test_data.drop(columns=["member_id", "signup_date"])
    test_uplift = predict_uplift(model_treat, model_control, X_test)
    test_data["uplift_score"] = test_uplift
    return test_data[["member_id", "uplift_score"]].sort_values(by="uplift_score", ascending=False)


def get_users(n_frac=0.2, param_path="best_params.json") -> pd.DataFrame:
    """
    Get the top n_frac of users with the highest uplift scores from the test set and save them to a CSV file.
    :param n_frac: fraction of users to select based on uplift scores (default is 0.2 for top 20%)
    :param param_path: path to the JSON file containing the best parameters for treatment and control models.
    :return: DataFrame with member_id, rank, and priority_score for the selected users.
    """
    train_data = build_features(dataset="train")
    json_path = os.path.join(param_path)

    with open(json_path, "r") as f:
        best_params = json.load(f)
        treatment_model = XGBClassifier(**best_params["treatment"])
        control_model = XGBClassifier(**best_params["control"])

    cv_score = cross_validate(train_data, n_splits=5, model=XGBClassifier,
                              t_param=best_params["treatment"], c_param=best_params["control"])


    control_model, treatment_model, _, _, _ = train(train_data,val_data=None,
                                                    treatemnt_model=treatment_model, control_model=control_model,
                                                    treatmeant_col="outreach", target_col="churn")
    test_data = build_features(dataset="test")
    test_results = inference(test_data, treatment_model, control_model)
    test_results = test_results.reset_index(drop=True)
    test_results["rank"] = test_results["uplift_score"].index + 1
    test_results["priority_score"] = test_results["uplift_score"].round(4)
    n = int(len(test_results) * n_frac)
    test_results = test_results[["member_id", "rank", "priority_score"]].head(n)
    pd.DataFrame.to_csv(test_results, "outreach_users.csv", index=False)
    return test_results



users = get_users()
