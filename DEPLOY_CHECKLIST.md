# Deployment Checklist: Streamlit Community Cloud

This checklist deploys the app for free. No credit card or paid service is needed. Do not continue if a payment method is requested.

## 1. Confirm the app runs locally

```bash
cd ml-dashboard
source .venv/bin/activate
streamlit run app.py
```

- [ ] The app opens at `http://localhost:8501`.
- [ ] The classification sample trains with **SVM**.
- [ ] The regression sample trains with **SVR**.
- [ ] The model and metrics downloads work.

## 2. Create the GitHub repository

1. Sign in to <https://github.com>.
2. Click **New repository**.
3. Name the repository, for example `ml-dashboard`.
4. Choose **Public** or **Private**. Streamlit Community Cloud supports both.
5. Do not add a README, `.gitignore`, or license if this folder already contains them.
6. Click **Create repository**.

## 3. Push the project

Run these commands from inside the `ml-dashboard` folder. Replace `<your-username>` and `<repository-name>`.

```bash
git init
git add .
git commit -m "Initial ML dashboard"
git branch -M main
git remote add origin https://github.com/<your-username>/<repository-name>.git
git push -u origin main
```

- [ ] `.venv/` is not committed.
- [ ] `app.py`, `models.py`, `pipeline.py`, `requirements.txt`, `.streamlit/config.toml`, and `data/` are committed.
- [ ] The repository opens on GitHub and shows the project files.

## 4. Sign in to Streamlit Community Cloud

1. Open <https://share.streamlit.io>.
2. Click **Sign in with GitHub**.
3. Authorize Streamlit to access your GitHub account.
4. Confirm that no payment information is requested.

- [ ] Signed in successfully.
- [ ] No payment method requested.

## 5. Create the app

1. Click **Create app** or **New app**.
2. Choose **Deploy a public app from GitHub** if prompted.
3. Select the repository you created.
4. Select the branch `main`.
5. Set the main file path to:

```text
app.py
```

6. Optionally set a custom app URL.
7. If an advanced Python-version option is shown, choose **Python 3.12**.
8. Click **Deploy**.

- [ ] Repository selected.
- [ ] Branch is `main`.
- [ ] Main file is `app.py`.
- [ ] Deployment started.

## 6. Confirm the free tier

- [ ] No credit card was requested.
- [ ] No paid plan was selected.
- [ ] The app started successfully.
- [ ] The app settings or resource panel show the memory limit. Write it down here:

```text
Memory limit shown: ______________________
```

The app is designed for a 1 GB budget because free-tier memory can vary.

## 7. Smoke test the deployed app

- [ ] The page loads without an error.
- [ ] **Use sample dataset** loads the classification data.
- [ ] The target defaults to `churn`.
- [ ] **SVM** with `kernel=rbf` and `C=1.0` trains and shows accuracy, precision, recall, F1, the confusion matrix, and the ROC curve.
- [ ] Switch to the regression sample, choose **SVR**, and train. R-squared, MAE, RMSE, and the predicted-vs-actual plot appear.
- [ ] Download the trained model and the metrics CSV.
- [ ] Upload a CSV with missing values and categorical columns and confirm it trains.
- [ ] Select enough grid-search values to exceed 50 combinations and confirm the clear error appears and the train button is disabled.

## 8. After deployment

- [ ] Copy the public app URL and submit it with the assignment.
- [ ] Note that the app sleeps after 12 hours of inactivity. Open the URL to wake it.
- [ ] If a deployment fails, open the app logs in Streamlit Community Cloud and check for a missing package or a wrong main-file path.
- [ ] Do not add secrets. This app needs none.

## Troubleshooting

- **App will not start:** Confirm the main file is `app.py` and that `requirements.txt` is in the repository root.
- **Package installation fails:** Confirm the repository uses the same Python version as the pins. Python 3.12 was used locally.
- **Memory error:** Use a smaller CSV, keep the built-in 50,000-row sample limit, and avoid huge one-hot encoded feature spaces.
- **Slow training:** Reduce the number of rows, use simpler hyperparameters, or disable the optional grid search.
