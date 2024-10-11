import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
import joblib
import io
import numpy as np

# Set Streamlit page configuration
st.set_page_config(layout="wide", page_title="Hierarchical Item Categorization")

# Constants
CSV_URL = "https://drive.google.com/uc?id=1cnau3XSlOjG4m9RZyk5UXTVakwPfuori&export=download"
REQUIRED_COLUMNS = ['Product Title', 'Category', 'Subcategory', 'Part Terminology ID - Name']

# Utility Functions
@st.cache_data
def load_data(url):
    try:
        data = pd.read_csv(url)
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

@st.cache_resource
def train_category_model(X, y):
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1, 2))),
        ('svm', LinearSVC(C=1.0))
    ])
    pipeline.fit(X, y)
    return pipeline

@st.cache_resource
def train_subcategory_models(training_data):
    subcat_models = {}
    for category in training_data['Category'].unique():
        category_data = training_data[training_data['Category'] == category]
        X_subcat_train = category_data['Product Title']
        y_subcat_train = category_data['Subcategory']
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1, 2))),
            ('svm', LinearSVC(C=1.0))
        ])
        pipeline.fit(X_subcat_train, y_subcat_train)
        subcat_models[category] = pipeline
    return subcat_models

@st.cache_resource
def train_part_terminology_models(training_data):
    part_term_models = {}
    for subcategory in training_data['Subcategory'].unique():
        subcat_data = training_data[training_data['Subcategory'] == subcategory]
        X_part_term_train = subcat_data['Product Title']
        y_part_term_train = subcat_data['Part Terminology ID - Name']
        if y_part_term_train.nunique() > 1:
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1, 2))),
                ('svm', LinearSVC(C=1.0))
            ])
            pipeline.fit(X_part_term_train, y_part_term_train)
            part_term_models[subcategory] = pipeline
        else:
            part_term_models[subcategory] = y_part_term_train.unique()[0]
    return part_term_models

def hierarchical_prediction(item_description, category_pipeline, subcat_models, part_term_models):
    try:
        predicted_category = category_pipeline.predict([item_description])[0]
    except Exception:
        predicted_category = 'Error in Category Prediction'
    
    # Step 2: Predict Subcategory based on Category and return top 3 predictions
    try:
        if predicted_category in subcat_models:
            subcat_model = subcat_models[predicted_category]
            decision_scores = subcat_model.decision_function([item_description])
            top_3_subcategories_indices = np.argsort(decision_scores[0])[-3:][::-1]
            top_3_subcategories = subcat_model.classes_[top_3_subcategories_indices]
        else:
            top_3_subcategories = ['Unknown Subcategory']
    except Exception:
        top_3_subcategories = ['Error in Subcategory Prediction']
    
    # Ensure we have 3 subcategories
    while len(top_3_subcategories) < 3:
        top_3_subcategories = np.append(top_3_subcategories, 'N/A')
    
    # Step 3: Predict top 3 Part Terminologies based on the top subcategory
    predicted_subcategory = top_3_subcategories[0]  # Use the top predicted subcategory
    try:
        if predicted_subcategory in part_term_models:
            if isinstance(part_term_models[predicted_subcategory], str):
                top_3_part_terms = [part_term_models[predicted_subcategory]] * 3  # If there's only one class
            else:
                part_term_model = part_term_models[predicted_subcategory]
                decision_scores_part_term = part_term_model.decision_function([item_description])
                top_3_part_term_indices = np.argsort(decision_scores_part_term[0])[-3:][::-1]
                top_3_part_terms = part_term_model.classes_[top_3_part_term_indices]
        else:
            top_3_part_terms = ['Unknown Part Terminology']
    except Exception:
        top_3_part_terms = ['Error in Part Terminology Prediction']
    
    # Ensure we have 3 part terminologies
    while len(top_3_part_terms) < 3:
        top_3_part_terms = np.append(top_3_part_terms, 'N/A')
    
    # Return top 3 subcategories and top 3 part terminologies
    return top_3_subcategories[0], top_3_subcategories[1], top_3_subcategories[2], top_3_part_terms[0], top_3_part_terms[1], top_3_part_terms[2]

def generate_template():
    template_df = pd.DataFrame(columns=['Item Number', 'Description'])
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        template_df.to_excel(writer, index=False)
    buffer.seek(0)
    return buffer

# Load and preprocess data
training_data = load_data(CSV_URL)

# Check for required columns
missing_columns = [col for col in REQUIRED_COLUMNS if col not in training_data.columns]
if missing_columns:
    st.error(f"Training data is missing the following columns: {', '.join(missing_columns)}")
    st.stop()

# Handle missing values
training_data['Product Title'] = training_data['Product Title'].fillna('')
training_data = training_data.dropna(subset=['Category', 'Subcategory', 'Part Terminology ID - Name'])

# Train models
with st.spinner("Training Category model..."):
    category_pipeline = train_category_model(training_data['Product Title'], training_data['Category'])

with st.spinner("Training Subcategory models..."):
    subcat_models = train_subcategory_models(training_data)

with st.spinner("Training Part Terminology models..."):
    part_term_models = train_part_terminology_models(training_data)

# Streamlit App UI
st.title("📦 Hierarchical Item Categorization")

st.markdown("""
This application categorizes items hierarchically into **Subcategory** and **Part Terminology** based on their descriptions.
You can either manually enter item descriptions or upload an Excel file for batch predictions.
""")

# Option 1: Manual Entry
st.header("🔹 Option 1: Manual Entry")
item_input = st.text_area("Enter item descriptions (one per line):", height=150)
items = [item.strip() for item in item_input.split("\n") if item.strip()]

if st.button("Get Hierarchical Predictions for Manual Entry"):
    if items:
        with st.spinner("Processing..."):
            df_manual = pd.DataFrame({'Item': items})
            predictions = df_manual['Item'].apply(
                lambda x: pd.Series(hierarchical_prediction(x, category_pipeline, subcat_models, part_term_models))
            )
            # Assign the predictions to the correct columns
            df_manual[['Predicted Subcategory 1', 'Predicted Subcategory 2', 'Predicted Subcategory 3', 
                       'Predicted Part Terminology 1', 'Predicted Part Terminology 2', 'Predicted Part Terminology 3']] = predictions
            st.success("### Predicted Subcategories and Part Terminologies")
            st.dataframe(df_manual)
            
            # Export as CSV
            csv_manual = df_manual.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Predictions as CSV",
                data=csv_manual,
                file_name="hierarchical_predictions_manual.csv",
                mime="text/csv",
                key='download-csv-manual'
            )
    else:
        st.warning("⚠️ Please enter at least one item description before submitting.")

# Divider
st.markdown("---")

# Option 2: File Upload
st.header("🔹 Option 2: Upload Excel File")

st.subheader("📄 Download Excel Template")
st.download_button(
    label="Download Template",
    data=generate_template(),
    file_name="hierarchical_categorization_template.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

st.subheader("📤 Upload Your Excel File")
uploaded_file = st.file_uploader("Upload an Excel file with 'Item Number' and 'Description' columns", type="xlsx")

if uploaded_file is not None:
    try:
        input_data = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error reading Excel file: {e}")
        st.stop()
    
    required_upload_cols = ['Item Number', 'Description']
    missing_upload_cols = [col for col in required_upload_cols if col not in input_data.columns]
    
    if missing_upload_cols:
        st.warning(f"The uploaded file is missing the following columns: {', '.join(missing_upload_cols)}")
    else:
        with st.spinner("Processing..."):
            predictions = input_data['Description'].apply(
                lambda x: pd.Series(hierarchical_prediction(x, category_pipeline, subcat_models, part_term_models))
            )
            input_data[['Predicted Subcategory 1', 'Predicted Subcategory 2', 'Predicted Subcategory 3', 
                        'Predicted Part Terminology 1', 'Predicted Part Terminology 2', 'Predicted Part Terminology 3']] = predictions
            st.success("### Predicted Subcategories and Part Terminologies")
            st.dataframe(input_data)
            
            # Export as CSV
            csv_excel = input_data.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Predictions as CSV",
                data=csv_excel,
                file_name="hierarchical_predictions_excel.csv",
                mime="text/csv",
                key='download-csv-excel'
            )
else:
    st.info("ℹ️ Please upload an Excel file to get started.")
