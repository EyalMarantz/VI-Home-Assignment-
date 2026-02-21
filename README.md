# VI-Home-Assignment-


Overview

This repository contains a reproducible end-to-end solution for identifying members who are most likely to benefit from outreach in order to reduce churn.

Rather than predicting churn alone, this solution estimates the incremental impact of outreach (uplift modeling).
The objective is to prioritize outreach toward members where it produces measurable behavioral change.

Methodology
Modeling Framework

We model this as an uplift learning problem using a T-Learner approach:

Train one model on treated (outreach) members

Train one model on control (no outreach) members

Estimate uplift per member as:

Uplift = P(churn | control) − P(churn | treatment)

If uplift > 0 → outreach reduces churn for that member.

Models used:

XGBoost classifiers

5-fold stratified cross-validation


If uplift > 0 → outreach reduces churn for that member.

Models used:

XGBoost classifiers

5-fold stratified cross-validation
