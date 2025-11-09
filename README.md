
# Telco Customer Churn Prediction (End-to-End Production ML on AWS)

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Production%20API-green)
![Docker](https://img.shields.io/badge/Containerized-Docker-blueviolet)
![AWS](https://img.shields.io/badge/Deploy-ECS%20Fargate-orange)
![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-yellow)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

**Production-grade Telecom Customer Churn Prediction System with FastAPI, Docker, MLflow, and AWS ECS Fargate Deployment.**

This project is a **real-world scalable production machine learning system** that predicts which telecom customers are likely to churn.  
Unlike typical notebook-only churn repos, this project includes full **MLOps pipeline, containerization, CI/CD and cloud deployment**.

> ### Business Impact
> In telecom, retaining customers is far more profitable than acquiring new ones.  
> Reducing churn by **even 5%** can increase profits by **20–40%**.  
> This system demonstrates how machine learning can directly generate measurable financial impact by enabling proactive retention campaigns.

---

## Key Features ✨

- Predict **customer churn probability**
- FastAPI based REST inference service (`/predict`)
- Gradio UI at `/ui` for business demo / validation
- MLflow experiment tracking + metric versioning
- Fully containerized using Docker
- CI/CD with GitHub Actions → automated build → deploy
- Production deployment on **AWS ECS Fargate behind ALB**
- CloudWatch logs for observability
- Health endpoint for ALB target group checks

---

## Tech Stack 🧰

| Layer | Tools |
|------:|:------|
| Modeling / Training | Python, scikit-learn, MLflow |
| Serving API | FastAPI |
| User Interface | Gradio |
| Containerization | Docker |
| Cloud Deploy | AWS ECS Fargate + ALB |
| CI/CD | GitHub Actions |
| Monitoring | AWS CloudWatch |

---

## Repository structure 📁

Top-level layout (clean tree):

```
Telco-churn-project/
├── src/                 # API, inference, model serving, preprocessing, features
├── models/              # trained model artifacts (if present)
├── notebooks/           # EDA and experimentation notebooks
├── scripts/             # utility scripts (data prep, training helpers)
├── mlruns/              # MLflow local experiment logs (if used)
├── data/                # raw and processed datasets
│   ├── raw/
│   └── processed/
├── requirements.txt
└── README.md
```

---

## How to run locally ▶️

1. Clone the repository:

```bash
git clone https://github.com/ShripadJagtap/Telco-churn-project.git
cd Telco-churn-project
```

2. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # macOS / Linux (zsh)
pip install --upgrade pip
pip install -r requirements.txt
```

3. Start the FastAPI app for development:

```bash
uvicorn src.app:app --reload
```

4. Open the Gradio UI in your browser:

http://127.0.0.1:8000/ui

---

## API endpoints 🔌

- POST `/predict`  — returns churn probability for a customer payload
- GET `/health`    — simple health check for ALB / orchestrator
- GET `/ui`        — Gradio-based demo UI

(Refer to `src/app` for precise request/response JSON schemas.)

---

## Deployment 🚀

Deployment architecture (ASCII diagram):

```
					Internet
					   |
					   v
				   [ALB / DNS]
					   |
		   -----------------------------
		   |                           |
		   v                           v
	[ECS Fargate Service]         [MLflow Tracking Server]
	(FastAPI + Gradio)                (optional)
		   |
	----------------
	|              |
	v              v
[CloudWatch]    [EFS / S3 (artifacts & logs)]

```

Notes:
- Build a Docker image for the FastAPI service, push to ECR, then deploy to ECS Fargate behind an Application Load Balancer (ALB).
- Use CloudWatch for logs and metrics; optionally use MLflow server (with backing storage) for experiment tracking.

---

## Model card (summary) 🧾

**Purpose**

Predict which telecom customers are likely to churn in the near future. Useful for targeting retention campaigns and prioritizing outreach.

**Intended use**

- Customer retention strategy and ROI-driven retention campaigns
- CRM automation and personalization
- Not intended for: credit/loan decisions, legal eligibility, or other high-stakes automated decisions without additional governance

**Training data**

Public Telco churn dataset containing contract type, billing, tenure, services subscribed, payment methods and monthly charges. Target is `Churn` (binary).

**Model type**

Scikit-learn binary classifier (best model selected via MLflow experiments).

**Key metrics (typical ranges)**

- ROC-AUC: ~0.80–0.85 (varies by experiment)
- Recall prioritized to reduce false negatives (hidden churners)
- Precision tuned to balance retention cost vs. false-positive outreach

**Ethical considerations**

- Avoid using sensitive demographic features without fairness checks
- Monitor model performance over time for data drift and fairness drift

**Versioning & reproducibility**

MLflow is used for experiment tracking and artifact reproducibility. Check `mlruns/` and the training scripts in `src/models`.

---

## Roadmap / future work 🛣️

- Prometheus metrics endpoint
- Cost-optimized decision thresholding
- Automated data-drift detection and scheduled retraining
- Export to ONNX / skops for portable inference
- Use AWS Secrets Manager for secrets instead of plain env vars

---

## Quick notes 📝

- If you plan to deploy to AWS, the repository assumes you will build a Docker image and push it to ECR, then deploy to ECS Fargate behind an Application Load Balancer.
- Check the `.github/workflows` directory for CI/CD examples (build, test, push, deploy).

---

## License ⚖️

MIT

## Author 👤

Shripad Jagtap
