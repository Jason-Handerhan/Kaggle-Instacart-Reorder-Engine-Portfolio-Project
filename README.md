---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.3
  kernelspec:
    display_name: Python 3 (ipykernel) * (Local)
    language: python
    name: micromamba-env-jupyterlab-py
---

# 🛒 Instacart Reorder Engine Project Overview

This project leverages the Kaggle Instacart Market Basket Analysis dataset to build a machine learning-powered recommendation engine designed to populate a personalized, 5-slot reorder widget during active shopping sessions. Delivering relevant product suggestions drives profit lift through three key mechanisms: increasing Average Order Value (AOV) by encouraging supplementary purchases, boosting conversion rates via reduced friction, and mitigating customer churn through saved time and increased engagement.

---

## 🌐 Full Documentation
#### 👉 **[Launch Full README](https://Jason-Handerhan.github.io/Kaggle-Instacart-Reorder-Engine-Portfolio-Project/)**

---

## 📌 Executive Highlights
* **Champion Architecture:** Rank Reciprocal Fusion (RRF) ensemble blending XGBoost & LightGBM Rankers & Binary Classifiers using Optuna-optimized weights.
* **Predictive Lift:** Achieved a **0.3674 Recall@5** on unseen test data—a **39.3% lift** over the prior-cart heuristic baseline (0.2638).
* **Financial Impact:** Projected **\$118M Annual Profit Lift** over a "no widget" scenario & **\$33.5M Lift** over a heuristic baseline (using SEC filings metrics for baseline assumptions).

## 🛠️ Tech Stack
* **Cloud Warehouse & Pipelines:** Google Cloud Storage, GCP BigQuery & Dataform (Medallion Architecture SQL: Bronze &rarr; Silver &rarr; Gold)
* **Machine Learning & MLOps:** Vertex AI Workbench (16 vCPU|64GB RAM|NVIDIA L4 GPU), XGBoost, LightGBM, Optuna (Bayesian Optimization), MLflow, SHAP
* **Business Intelligence:** Power BI (`.pbip` CI/CD format & downloadable `.pbix`), Star Schema Semantic Model, DAX
* **SDLC, Infrastructure Provisioning, & Version Control:** Google Cloud Shell, GCS, IAM, GitHub (`dev` / `main` branching strategy)

---

## 📂 Repository Structure
```text
├── Notebooks/              # EDA and evaluation notebooks
├── config/                 # GCP service API specifications & environment parameters
├── dashboard/              # .pbip & .pbix versions of power bi dashboard
├── definitions/            # Dataform medallion ELT SQL transformation models
├── images/                 # Images used in full .html documentation
├── scripts/                # Infrastructure setup, VM deployment, & ml pipeline execution shell scripts
├── src/                    # Python training & Bayesian Search Tuning scripts, & RRF ensemble blending weight optimization 
├── index.html              # Live documentation source page
├── requirements.txt        # Environment dependencies 
└── workflow_settings.yaml  # Dataform configuration file