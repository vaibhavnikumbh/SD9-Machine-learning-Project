# SDG 9: Road Infrastructure Progress Analysis Using Machine Learning

## Project overview
This beginner-friendly project explores road-construction progress using K-Means clustering and Random Forest classification. It supports UN Sustainable Development Goal 9: Industry, Innovation and Infrastructure.

## Columns
- `project_id`: demonstration project identifier
- `road_length_km`: road length in kilometres
- `estimated_cost_crore`: estimated cost in ₹ crore
- `planned_duration_months`: planned duration
- `elapsed_months`: months elapsed
- `physical_progress_pct`: physical completion percentage
- `expenditure_crore`: expenditure in ₹ crore
- `delay_months`: elapsed time beyond planned duration, floored at zero
- `progress_category`: Low (<40%), Moderate (40–74.9%), High (75%+), derived from physical progress

## Models
1. **K-Means clustering** groups projects by their numeric characteristics. The silhouette score describes cluster separation; it is not accuracy.
2. **Random Forest classifier** predicts the progress category for a demonstration task. Because the labels are derived from synthetic progress values, results are only a pipeline demonstration and do not establish real-world predictive performance.

## Run locally
Install Python 3.10+ then run:
```bash
pip install pandas numpy matplotlib scikit-learn
python sdg9_road_progress_ml.py
```
Keep the CSV and Python file in the same folder.

## Run in Google Colab
1. Upload both files to Colab, or upload the ZIP and extract it.
2. Run `!pip install pandas numpy matplotlib scikit-learn`
3. Run the script after the CSV is in the current working directory.

## Limitations and future scope
- Data is synthetic and simplified.
- A single snapshot does not establish whether a project will finish late or exceed budget.
- Future work: use official PMGSY records, repeated monthly progress snapshots, weather/terrain/contract information, and time-based validation to forecast delays or costs responsibly.


## Deploy as a live web app (Streamlit Community Cloud)
1. Upload `app.py`, `road_construction_sample_data.csv`, `requirements.txt`, and `README.md` to a GitHub repository.
2. Visit https://share.streamlit.io/ and sign in with GitHub.
3. Choose **Create app**, select your repository and branch, and set the main file path to `app.py`.
4. Click **Deploy**. After it finishes, copy the `*.streamlit.app` URL and share it with your supervisor.
5. Keep the repository public if the reviewer should be able to inspect the code without signing in.

The app uses synthetic sample data. Clearly disclose this in the app and report.
