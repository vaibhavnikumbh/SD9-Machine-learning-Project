import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

st.set_page_config(page_title="SDG 9 | Road Infrastructure ML", page_icon="🛣️", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("road_construction_sample_data.csv")

@st.cache_resource
def train_models(df):
    cluster_features = [
        "road_length_km", "estimated_cost_crore", "planned_duration_months",
        "elapsed_months", "physical_progress_pct", "expenditure_crore", "delay_months"
    ]
    cluster_model = make_pipeline(
        StandardScaler(),
        KMeans(n_clusters=3, random_state=42, n_init=10)
    )
    cluster_model.fit(df[cluster_features])
    df_out = df.copy()
    df_out["cluster"] = cluster_model.predict(df[cluster_features])

    features = [
        "road_length_km", "estimated_cost_crore", "planned_duration_months",
        "elapsed_months", "expenditure_crore", "delay_months"
    ]
    X_train, X_test, y_train, y_test = train_test_split(
        df[features], df["progress_category"], test_size=0.25,
        random_state=42, stratify=df["progress_category"]
    )
    clf = RandomForestClassifier(n_estimators=150, random_state=42, class_weight="balanced")
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return df_out, clf, features, X_test, y_test, predictions, accuracy

st.title("🛣️ Road Infrastructure Progress Analysis")
st.subheader("Machine Learning prototype aligned with UN Sustainable Development Goal 9")
st.info("Dataset notice: This app uses synthetic demonstration data created for this project. It is not official government data and results should not be interpreted as real-world performance.")

df = load_data()
clustered, model, features, X_test, y_test, predictions, accuracy = train_models(df)

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio("Choose a section", [
    "Project Overview", "Explore Dataset", "Progress Analysis", "ML Model Demo", "Methodology & Limitations"
])

if page == "Project Overview":
    st.markdown("This educational prototype explores road-project characteristics with K-Means clustering and demonstrates progress-category classification with a Random Forest model.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sample projects", f"{len(df):,}")
    c2.metric("Average physical progress", f"{df['physical_progress_pct'].mean():.1f}%")
    c3.metric("Average estimated cost", f"₹{df['estimated_cost_crore'].mean():.1f} Cr")
    c4.metric("Demo test accuracy", f"{accuracy*100:.1f}%")
    st.markdown("### Progress category distribution")
    counts = df["progress_category"].value_counts().reindex(["Low", "Moderate", "High"]).fillna(0)
    fig, ax = plt.subplots()
    counts.plot(kind="bar", ax=ax)
    ax.set_xlabel("Progress category")
    ax.set_ylabel("Number of projects")
    ax.set_title("Sample Road Project Categories")
    st.pyplot(fig)
    st.markdown("### SDG 9 connection")
    st.write("SDG 9 promotes resilient infrastructure and innovation. Data analysis can help demonstrate ways project teams might review progress and identify patterns. This prototype does not prove an improvement in actual infrastructure outcomes.")

elif page == "Explore Dataset":
    st.markdown("### Sample project records")
    categories = st.multiselect("Filter progress category", ["Low", "Moderate", "High"], default=["Low", "Moderate", "High"])
    filtered = df[df["progress_category"].isin(categories)]
    st.dataframe(filtered, use_container_width=True, hide_index=True)
    st.caption(f"Showing {len(filtered)} of {len(df)} synthetic sample records.")
    st.download_button("Download filtered CSV", filtered.to_csv(index=False).encode("utf-8"), "road_projects_filtered.csv", "text/csv")

elif page == "Progress Analysis":
    st.markdown("### Project progress versus elapsed time")
    fig, ax = plt.subplots()
    scatter = ax.scatter(clustered["elapsed_months"], clustered["physical_progress_pct"], c=clustered["cluster"])
    ax.set_xlabel("Elapsed months")
    ax.set_ylabel("Physical progress (%)")
    ax.set_title("K-Means project clusters")
    st.pyplot(fig)
    st.markdown("### Average metrics by cluster")
    summary = clustered.groupby("cluster")[["physical_progress_pct", "estimated_cost_crore", "expenditure_crore", "delay_months"]].mean().round(2)
    st.dataframe(summary, use_container_width=True)
    st.caption("Cluster numbers are arbitrary group labels, not rankings. K-Means discovers similarity patterns; it does not predict future completion.")

elif page == "ML Model Demo":
    st.markdown("### Enter hypothetical project details")
    st.caption("The model predicts a category from the synthetic training data. It is for demonstration only.")
    with st.form("prediction_form"):
        a, b = st.columns(2)
        with a:
            length = st.number_input("Road length (km)", min_value=0.1, max_value=200.0, value=12.0)
            cost = st.number_input("Estimated cost (₹ crore)", min_value=0.1, max_value=1000.0, value=18.0)
            planned = st.number_input("Planned duration (months)", min_value=1.0, max_value=120.0, value=12.0)
        with b:
            elapsed = st.number_input("Elapsed months", min_value=0.0, max_value=120.0, value=6.0)
            expenditure = st.number_input("Expenditure so far (₹ crore)", min_value=0.0, max_value=1000.0, value=7.5)
            delay = st.number_input("Delay months", min_value=0.0, max_value=120.0, value=0.0)
        submitted = st.form_submit_button("Classify project progress")
    if submitted:
        row = pd.DataFrame([{
            "road_length_km": length,
            "estimated_cost_crore": cost,
            "planned_duration_months": planned,
            "elapsed_months": elapsed,
            "expenditure_crore": expenditure,
            "delay_months": delay
        }])[features]
        prediction = model.predict(row)[0]
        st.success(f"Predicted progress category: **{prediction}**")
        st.write("This is a model classification, not a reliable forecast. Real-world use requires verified project data and validation.")
    st.markdown("### Model evaluation on held-out sample rows")
    st.metric("Random Forest test accuracy", f"{accuracy*100:.1f}%")
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).transpose().round(3), use_container_width=True)
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(y_test, predictions, labels=["Low", "Moderate", "High"], ax=ax)
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)

else:
    st.markdown("### Methodology")
    st.write("1. Load the CSV using pandas.")
    st.write("2. Standardize numeric features and apply K-Means with three clusters.")
    st.write("3. Split labelled data into training and test sets.")
    st.write("4. Train a Random Forest classifier and report classification metrics.")
    st.markdown("### Important limitations")
    st.warning("The dataset is synthetic and progress categories are derived from the physical-progress percentage. Accuracy is illustrative, not evidence of real-world model quality. The app does not forecast final cost, delay, or completion date.")
    st.markdown("### Future scope")
    st.write("Replace the sample data with verified official records and, if available, repeated monthly project snapshots. Then test delay or cost-overrun forecasting with time-based validation.")
    st.markdown("### References")
    st.markdown("- [UN Sustainable Development Goal 9](https://sdgs.un.org/goals/goal9)")
    st.markdown("- [PMGSY online portal](https://pmgsy.dord.gov.in/dbweb)")
