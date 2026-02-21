import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
from sklearn.model_selection import train_test_split, ParameterSampler, KFold, StratifiedKFold
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import json
from warnings import filterwarnings

filterwarnings("ignore")


def predict_uplift(model_treat, model_control, X) -> np.ndarray:
    """
    Predict uplift scores for each user based on the treatment and control models.

    :param model_treat: trained model for the treatment group (predicts probability of churn if treated).
    :param model_control: trained model for the control group (predicts probability of churn if not treated).
    :param X: Feature matrix for the users to predict uplift for.
    :return: predicted uplift scores (control - treatment) for each user.
    """
    p_treat = model_treat.predict_proba(X)[:, 1]
    p_control = model_control.predict_proba(X)[:, 1]
    uplift = p_control - p_treat
    return uplift


def train(train_data: pd.DataFrame,
          val_data: pd.DataFrame = None,
          treatemnt_model = None,
          control_model = None,
          treatmeant_col: str ="outreach",
          target_col: str ="churn",
          k_frac=0.2):

    """
    :param train_data: training data containing features, treatment indicator, and target variable.
    :param val_data: validation data for evaluating uplift predictions (optional).
    :param treatemnt_model: model to predict outcomes for the treatment group.
    :param control_model: model to predict outcomes for the control group.
    :param treatmeant_col: treatment indicator column name in the dataset (e.g., "outreach").
    :param target_col: target variable column name in the dataset (e.g., "churn").
    :param k: fraction of top users to evaluate uplift on (default is 0.2 for top 20%).
    :return: trained treatment and control models, validation uplift predictions, uplift at k, and gain at k.
    """
    X, y, teatment = train_data.drop(columns=["member_id", "signup_date", treatmeant_col, target_col]),\
        train_data[target_col], train_data[treatmeant_col]

    treatemnt_model.fit(X[teatment == 1], y[teatment == 1])
    control_model.fit(X[teatment == 0], y[teatment == 0])
    if val_data is None:
        return treatemnt_model, control_model, None, None, None

    X_val, y_val, treatment_val = val_data.drop(columns=["member_id", "signup_date", treatmeant_col, target_col]),\
    val_data[target_col], val_data[treatmeant_col]

    validation_uplift = predict_uplift(treatemnt_model, control_model, X_val)
    uplift_at_k = evaluate_uplift(y_val, treatment_val, validation_uplift, k_frac=k_frac)
    gain_at_k = uplift_at_k * int(len(val_data) * k_frac)  # Total churn prevented proxy for top k users
    return treatemnt_model, control_model, validation_uplift, uplift_at_k, gain_at_k

def cross_validate(data: pd.DataFrame, n_splits: int = 5,
                   model = XGBClassifier,
                   t_param = {},
                   c_param ={},
                   k_frac=0.2,
                   placebo=False):
    """
    :param data: DataFrame containing features, treatment indicator, and target variable.
    :param n_splits: number of folds for cross-validation.
    :param model: machine learning model class to use for treatment and control (e.g., XGBClassifier or LogisticRegression).
    :param t_param: parameters for the treatment model (e.g., {"n_estimators": 100, "max_depth": 5}).
    :param c_param: parameters for the control model (e.g., {"n_estimators": 100, "max_depth": 5}).
    :param k_frac: fraction of top users to evaluate uplift on (default is 0.2 for top 20%).
    :param placebo: bool. If True, shuffle treatment assignment in training data to create a placebo test (should result in uplift close to 0).
    :return: CV average uplift at top k_frac users across folds.
    """
    strata = data["outreach"].astype(str) + "_" + data["churn"].astype(str)
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    uplift_scores = []
    gain_scores = []
    for train_index, val_index in kf.split(data, strata):
        train_data = data.iloc[train_index].copy()
        val_data = data.iloc[val_index].copy()

        if placebo:
            # Shuffle treatment assignment in training data to create a placebo test
            train_data["outreach"] = np.random.choice([0, 1], size=len(train_data))

        treatment_model = model(**t_param, random_state=42)
        control_model = model(**c_param, random_state=42)

        _, _, validation_uplift, uplift_at_k, _ = train(train_data, val_data,
                                        treatemnt_model=treatment_model, control_model=control_model,
                                       treatmeant_col="outreach", target_col="churn", k_frac=k_frac)
        uplift_scores.append(uplift_at_k)
    return np.array(uplift_scores).mean()

def evaluate_uplift(y_val, t_val, uplift_val, k_frac=0.2):
    """

    :param y_val: values of the target variable (e.g., churn) for the validation set.
    :param t_val: values of the treatment indicator (e.g., outreach) for the validation set.
    :param uplift_val: uplift scores predicted by the model for the validation set.
    :param k_frac: fraction of top users to evaluate uplift on (default is 0.2 for top 20%).
    :return:
    """
    df = pd.DataFrame({
        "y": y_val,
        "t": t_val,
        "uplift": uplift_val
    }).sort_values("uplift", ascending=False)

    k = int(len(df) * k_frac)
    top = df.iloc[:k]

    treated = top[top["t"] == 1]["y"]
    control = top[top["t"] == 0]["y"]

    if len(treated) == 0 or len(control) == 0:
        return 0

    observed_uplift = control.mean() - treated.mean()
    return observed_uplift


def model_selection(train_data: pd.DataFrame,
                    num_iters=100,
                    experiment_name="xgb_base_features",
                    placebo=False):

    """
    Use for model selection and hyperparameter tuning. Randomly samples pairs of treatment/control model parameters,
    evaluates CV uplift, and keeps track of the best performing pair. If placebo=True, performs a placebo test
    by shuffling treatment assignment in training data (should result in uplift close to 0).

    :param train_data:
    :param num_iters:
    :param experiment_name:
    :param placebo:
    :return: score of best performing model pair (CV average uplift at top 20% across folds).
    """

    best_score = -np.inf
    best_params = {"treatment": None, "control": None, "score": None}

    param_dist = {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 5, 7, 10],
        "learning_rate": [0.01, 0.1, 0.2],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "enable_categorical": [True]
    }

    param_sampler = list(ParameterSampler(param_dist, n_iter=1000, random_state=42))

    for _ in tqdm(range(num_iters)):
        t_param, c_param = np.random.choice(param_sampler, 2, replace=False)
        cv_score = cross_validate(train_data, n_splits=5, model = XGBClassifier,
                                  t_param=t_param, c_param=c_param, placebo=placebo)
        if cv_score > best_score:
            best_score = cv_score
            best_params["treatment"] = t_param
            best_params["control"] = c_param
            best_params["score"] = str(cv_score)
    print(f"Best CV Uplift at top 20%: {best_score:.4f} with params: {best_params}")
    json_path = os.path.join("experiments", f"{experiment_name}_best_params.json")

    with open(json_path, "w") as f:
        json.dump(best_params, f)

    return best_score
def get_gain_curve(train_data, best_params: dict, user_step=0.05) -> pd.DataFrame:
    """
    :param train_data: training data containing features, treatment indicator, and target variable.
    :param best_params: parameters for the treatment and control models
    :param user_step: step size for users to evaluate gain at (e.g., 0.05 for every 5% of users).
    :return: cumulative gain at each user step as a DataFrame
    """
    user_step = int(len(train_data) * user_step)
    Ns = np.arange(users_step, len(train_data) + users_step, step=users_step)
    ks = [n / len(train_data) for n in Ns]

    gain_at_k = []

    for n, k_frac in tqdm(zip(Ns, ks)):
        cv_uplift = cross_validate(
            train_data, n_splits=5, model=XGBClassifier,
            t_param=best_params["treatment"], c_param=best_params["control"],
            k_frac=k_frac)
        gain = cv_uplift * n  # Total churn prevented proxy for top n users
        gain_at_k.append(gain)

    curve = pd.DataFrame({"N": Ns, "k_frac": ks, "gain_at_k": gain_at_k})
    pd.DataFrame.to_csv(curve, "gain_curve.csv", index=False)
    return curve

def choose_n(train_data: pd.DataFrame, run=False,
             param_path="xgb_base_features", outreach_cost=0.02) -> None:

    """
    Help choose optimal n for outreach by plotting gain@k and marginal gain@k curves.
    The optimal n is where marginal gain@k intersects outreach cost.

    :param train_data: training data containing features, treatment indicator, and target variable.
    :param run: bool. If True, runs the gain curve experiment to compute gain@k. If False, reads pre-computed gain curve from "gain_curve.csv".
    :param param_path: path to JSON file containing best parameters for treatment and control models (e.g., "xgb_base_features_best_params.json").
    :param outreach_cost: Assumed cost per outreach (e.g., 0.02 for 2% cost). Used to plot net gain and marginal gain curves.
    :return:
    """

    json_path = os.path.join("experiments", f"{param_path}_best_params.json")
    with open(json_path, "r") as f:
        best_params = json.load(f)

    if run:
        curve = get_gain_curve(users_step=500, train_data=train_data, best_params=best_params)
    else:
        curve = pd.read_csv("gain_curve.csv")

    plt.figure(figsize=(10, 6))
    fig, axs = plt.subplots(figsize=(10, 6), shape=(10, 6))


    plt.plot(curve["N"], curve["gain_at_k"],
             marker="o", label=f"No Cost")
    plt.plot(curve["N"], curve["gain_at_k"] - outreach_cost * curve["N"],
                marker="o", label=f"With Outreach Cost = {outreach_cost}")
    plt.title(f"Net Gain@k Estimate)")
    plt.xlabel("Users targeted (n)")
    plt.ylabel("Net Gain@k")
    plt.grid(True)
    plt.legend()
    plt.show()

    curve["marginal_gain_at_k"] = curve["gain_at_k"] - curve["gain_at_k"].shift(1, fill_value=0)
    curve["marginal_gain_at_k"] = curve["marginal_gain_at_k"] / (curve["N"] - curve["N"].shift(1, fill_value=0))
    plt.figure(figsize=(10, 6))
    plt.plot(curve["N"], curve["marginal_gain_at_k"],
             marker="o")
    plt.hlines(outreach_cost, xmin=0, xmax=curve["N"].max(), colors="red", linestyles="--", label="Cost Per Outreach")
    plt.title(f"Marginal Uplift@k Estimate)")
    plt.xlabel("Users targeted (n)")
    plt.ylabel("Marginal Uplift @k")
    plt.grid(True)
    plt.legend()
    plt.show()
