# VI-Home-Assignment-


2️⃣ Problem Overview

**Goal
Identify members most likely to benefit from outreach to reduce churn.

Rather than predicting churn alone, we estimate the incremental treatment effect (uplift) — the expected reduction in churn caused by outreach.


2️⃣ Method Overview
**Modeling Approach

We model the problem as an uplift learning task using a T-Learner framework:

Train one model on treated members

Train one model on control members

Uplift = P(churn | control) − P(churn | treatment)

We use:

XGBoost classifiers

5-fold stratified cross-validation

Gain@20% as the primary metric
