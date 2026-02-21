# 🎯 Predictive Uplift Modeling for Targeted Outreach

## 📌 Overview

This repository contains a solution for
identifying members who are most likely to benefit from outreach in
order to reduce churn.

Rather than predicting churn directly, this project models the
**incremental effect of outreach** using an uplift learning framework.

The objective is to prioritize outreach toward members where
intervention produces measurable behavioral change.

------------------------------------------------------------------------

# 🧠 Problem Formulation

An outreach event occurred between the observation period and the churn
measurement window.

We therefore model this as a treatment effect estimation problem:

-   Treatment: `outreach`
-   Outcome: `churn`
-   Goal: Estimate the **incremental reduction in churn** caused by
    outreach.

We define uplift as:

$$
\text{Uplift}(x) = P(\text{churn} \mid \text{control}, x) - P(\text{churn} \mid \text{treatment}, x)
$$

Positive uplift means outreach reduces churn for that member.

------------------------------------------------------------------------

# 🏗 Modeling Approach

## T-Learner Framework

We use a T-Learner architecture:

-   Train a model on treated members
-   Train a model on control members
-   Predict uplift as the difference of probabilities

$$
\hat{\tau}(x) = \hat{P}_{control}(x) - \hat{P}_{treatment}(x)
$$

Models used:
- XGBoost classifiers as a basic model
- 5-fold stratified cross-validation to ensure robustness

------------------------------------------------------------------------

# 🛠 Feature Engineering

Feature design focused on three core signals:

## 1️⃣ Engagement Intensity

-   App sessions
-   Active days
-   Web visits
-   Unique web pages
-   Claims count

## 2️⃣ Recency Signals

-   Days since last app activity
-   Days since last web visit
-   Days since last claim

## 3️⃣ Behavioral Content Signals

Based on web URL and title keyword buckets:

-   Nutrition
-   Activity
-   Sleep
-   Stress

## 4️⃣ Interaction 

Interaction between insurance claims and website content (e.g., diabetes × nutrition engagement)

Features were selected based on:
- Domain relevance
- Stability across CV folds
- Interpretability - Business alignment

------------------------------------------------------------------------

# 📊 Model Evaluation

Primary metric:

## Uplift@k

For top-k ranked members:

$$
\widehat{u}(k) =
\bar{Y}_{control, top-k} -
\bar{Y}_{treated, top-k}
$$

Cumulative gain is defined as:

$$
G(k) = \widehat{u}(k) \cdot k
$$

This represents the estimated number of churn events prevented if we
Contact the top-k members.

------------------------------------------------------------------------

## 🔬 Experiments

To validate the evaluation pipeline, we conducted:

-   Placebo test (shuffled treatment in training) to ensure model robustness.
-   Tested XG-Boost Model on subsets of features:
    - Basic Features: Engagement + Recovery 
    - Basic Features + Contened Features
    - Basic Feature + Content Features + Interactions

Feature selection and hyperparameter tuning were done using 5-fold cross-validation

### Results

## Model Performance (Uplift@20%)

| Model                                           | Uplift@20% |
|-------------------------------------------------|------------|
| Placebo Test                                    | ~2%        |
| XGBoost (Base Features)                         | ~5%        |
| XGBoost (Base + Context Features)               | **~7%**    |
| XGBoost (Base + Context + Interaction Terms)    | ~5%        |

The final model more than doubles uplift relative to baseline and
clearly exceeds the noise floor.

------------------------------------------------------------------------

# 📈 Selecting Outreach Size (n)

We evaluate cumulative gain curves across multiple targeting sizes:

$$
G(n) = \widehat{u}(n) \cdot n
$$

Marginal uplift between targeting levels:

$$
\text{Marginal}(n) =
\frac{G(n_i) - G(n_{i-1})}{n_i - n_{i-1}}
$$

We expand outreach while:

$$
\text{Marginal uplift} > \text{Cost / Value ratio}
$$

------------------------------------------------------------------------

## 💰 Economic Assumptions



Let:

-   $V$ = lifetime value per member\
-   $C$ = outreach cost per member

Outreach is profitable when:

$$
\widehat{u} > \frac{C}{V}
$$

Given:

-   Observed uplift ≈ 7%
-   Assumed cost/value ratio ≈ 2%

We retain a comfortable profitability margin.

------------------------------------------------------------------------

# 📁 Repository Structure

    .
    ├── experiments/
    │   ├── xgb_base_features_best_params.json
    |   ├── xgb_base_content_features_best_params.json
    |   ├── xgb_base_content_intersection_features_best_params.json
    |   ├── experiment_results.csv
    |   └── placebo_reults.csv
    ├── src
    │   ├── features.py
    │   ├── train_evaluate.py
    │   ├── inference.py
    │   └── main.py
    |── data.zip
    ├── outreach_users.csv
    ├── requirements.txt
    └── README.md

------------------------------------------------------------------------

# ⚙ Setup & Run Instructions

## 1. Install dependencies

``` bash
pip install -r requirements.txt
```

## 2. Unzip Data File

``` bash
unzip data.zip
```

## 3. Run the inference pipeline

``` bash
python inference.py
```

This will:

-   Train uplift models
-   Validate via cross-validation
-   Score test members
-   Generate final ranked output file

------------------------------------------------------------------------

# 📤 Final Output

The final submission file:

    outreach_users.csv

Contains:

-   `member_id`
-   `priority_score` (predicted uplift)
-   `rank`

Sorted from highest to lowest expected incremental impact.

------------------------------------------------------------------------

# ✅ Design Decisions Summary

-   Modeled outreach as a causal treatment effect\
-   Used stratified CV for balanced evaluation\
-   Validated robustness via placebo tests\
-   Selected outreach size using marginal economic return\
-   Focused on interpretable, business-aligned features

------------------------------------------------------------------------

# 🏁 Conclusion

This solution provides:

-   A causal uplift modeling framework\
-   Robust cross-validated evaluation\
-   Economically informed outreach selection\
-   A reproducible end-to-end pipeline

The final ranked list enables targeted intervention where outreach is
most likely to reduce churn and improve member outcomes.
