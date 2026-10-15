"""
SDG 9: Road Infrastructure Progress Analysis Using Machine Learning
IMPORTANT: The included CSV is synthetic demonstration data, not official project records.
Run: pip install pandas numpy matplotlib scikit-learn
     python sdg9_road_progress_ml.py
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import silhouette_score

DATA_FILE = "road_construction_sample_data.csv"

def main():
    df = pd.read_csv(DATA_FILE)
    print("\\nSDG 9: Road Infrastructure Progress Analysis")
    print(f"Rows: {len(df)}")
    print("\\nSample data:")
    print(df.head())
    print("\\nProgress category counts:")
    print(df["progress_category"].value_counts())

    # 1) Unsupervised learning: group projects by project size, time, and progress
    cluster_features = [
        "road_length_km", "estimated_cost_crore", "planned_duration_months",
        "elapsed_months", "physical_progress_pct", "expenditure_crore", "delay_months"
    ]
    cluster_pipe = make_pipeline(StandardScaler(), KMeans(n_clusters=3, random_state=42, n_init=10))
    df["cluster"] = cluster_pipe.fit_predict(df[cluster_features])
    scaled = cluster_pipe.named_steps["standardscaler"].transform(df[cluster_features])
    score = silhouette_score(scaled, df["cluster"])
    print(f"\\nK-Means silhouette score (descriptive, not predictive accuracy): {score:.3f}")
    print("\\nCluster summary:")
    print(df.groupby("cluster")[["physical_progress_pct","estimated_cost_crore","expenditure_crore","delay_months"]].mean().round(2))

    # 2) Supervised classification: classify progress category.
    # This is a demonstration classification task; category is defined from progress percentage.
    features = ["road_length_km", "estimated_cost_crore", "planned_duration_months",
                "elapsed_months", "expenditure_crore", "delay_months"]
    X = df[features]
    y = df["progress_category"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    model = RandomForestClassifier(n_estimators=150, random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    print(f"\\nRandom Forest test accuracy: {acc:.3f}")
    print("\\nClassification report:")
    print(classification_report(y_test, pred, zero_division=0))
    print("\\nNote: Since the data is synthetic and the category is derived from progress, these scores are for demonstration only.")

    # Visualizations
    plt.figure(figsize=(8,5))
    plt.scatter(df["elapsed_months"], df["physical_progress_pct"], c=df["cluster"])
    plt.xlabel("Elapsed months")
    plt.ylabel("Physical progress (%)")
    plt.title("Road Projects: Progress vs Elapsed Time (K-Means clusters)")
    plt.tight_layout()
    plt.savefig("progress_clusters.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7,4))
    df["progress_category"].value_counts().reindex(["Low","Moderate","High"]).plot(kind="bar")
    plt.xlabel("Progress category")
    plt.ylabel("Number of projects")
    plt.title("Distribution of Road Project Progress Categories")
    plt.tight_layout()
    plt.savefig("progress_categories.png", dpi=160)
    plt.close()

    ConfusionMatrixDisplay.from_predictions(y_test, pred, labels=["Low","Moderate","High"], cmap=None)
    plt.title("Random Forest: Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=160)
    plt.close()

    importances = pd.Series(model.feature_importances_, index=features).sort_values()
    plt.figure(figsize=(8,5))
    importances.plot(kind="barh")
    plt.xlabel("Feature importance")
    plt.title("Random Forest Feature Importance")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=160)
    plt.close()

    # Example prediction for a hypothetical new project
    example = pd.DataFrame([{
        "road_length_km": 12.0,
        "estimated_cost_crore": 18.0,
        "planned_duration_months": 12.0,
        "elapsed_months": 6.0,
        "expenditure_crore": 7.5,
        "delay_months": 0.0
    }])
    print("\\nExample hypothetical project prediction:")
    print("Inputs:", example.to_dict(orient="records")[0])
    print("Predicted progress category:", model.predict(example)[0])
    print("\\nGenerated charts: progress_clusters.png, progress_categories.png, confusion_matrix.png, feature_importance.png")

if __name__ == "__main__":
    main()
