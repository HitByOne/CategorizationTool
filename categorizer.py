import io
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")

# ==================== PAGE CONFIG ====================
st.set_page_config(
    layout="wide",
    page_title="Hierarchical Item Categorization",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "### Hierarchical Item Categorization System\nA machine learning-powered tool for intelligent product categorization."
    }
)

# ==================== CUSTOM STYLING ====================
st.markdown("""
<style>
    :root {
        --primary: #0066cc;
        --primary-light: #e6f2ff;
        --success: #00a86b;
        --warning: #ff9900;
        --danger: #cc0000;
        --dark: #1a1a1a;
        --light: #f8f9fa;
    }

    .stTabs [data-baseweb="tab-list"] button {
        font-weight: 600;
        font-size: 16px;
    }

    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #ffffff 100%);
        color: #1a1a1a;
        padding: 20px;
        border-radius: 12px;
        border-left: 4px solid var(--primary);
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }

    .metric-card div,
    .metric-card span,
    .metric-card p,
    .metric-card strong {
        color: #1a1a1a !important;
    }

    .success-box {
        background-color: #f0fdf4;
        border-left: 4px solid var(--success);
        color: #14532d;
        padding: 16px;
        border-radius: 8px;
        margin: 16px 0;
    }

    .warning-box {
        background-color: #fffbeb;
        border-left: 4px solid var(--warning);
        color: #78350f;
        padding: 16px;
        border-radius: 8px;
        margin: 16px 0;
    }

    .info-box {
        background-color: var(--primary-light);
        border-left: 4px solid var(--primary);
        color: #1e3a8a;
        padding: 16px;
        border-radius: 8px;
        margin: 16px 0;
    }

    .success-box *,
    .warning-box *,
    .info-box * {
        color: inherit !important;
    }

    .table-container {
        overflow-x: auto;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
    }
</style>
""", unsafe_allow_html=True)

# ==================== CONSTANTS & CONFIGURATION ====================
CSV_URL = "https://drive.google.com/uc?id=1cnau3XSlOjG4m9RZyk5UXTVakwPfuori&export=download"
REQUIRED_COLUMNS = ['Product Title', 'Category', 'Subcategory', 'Part Terminology ID - Name']
MAX_BATCH_SIZE = 5000

# ==================== SESSION STATE INITIALIZATION ====================
if 'training_complete' not in st.session_state:
    st.session_state.training_complete = False
if 'last_predictions' not in st.session_state:
    st.session_state.last_predictions = None

# ==================== UTILITY FUNCTIONS ====================
@st.cache_data
def load_data(url):
    try:
        data = pd.read_csv(url, on_bad_lines='skip')
        return data
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        st.stop()


@st.cache_resource
def train_category_model(X, y):
    try:
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=5000)),
            ('svm', LinearSVC(C=1.0, max_iter=2000, random_state=42))
        ])
        pipeline.fit(X, y)
        return pipeline
    except Exception as e:
        st.error(f"Error training category model: {e}")
        st.stop()


@st.cache_resource
def train_subcategory_models(training_data):
    subcat_models = {}
    categories = training_data['Category'].dropna().unique()

    for category in categories:
        category_data = training_data[training_data['Category'] == category]
        X_subcat = category_data['Product Title'].fillna('').astype(str)
        y_subcat = category_data['Subcategory'].astype(str)

        if y_subcat.nunique() > 1:
            try:
                pipeline = Pipeline([
                    ('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=5000)),
                    ('svm', LinearSVC(C=1.0, max_iter=2000, random_state=42))
                ])
                pipeline.fit(X_subcat, y_subcat)
                subcat_models[category] = pipeline
            except Exception as e:
                st.warning(f"Could not train model for category '{category}': {e}")
        else:
            subcat_models[category] = y_subcat.iloc[0]

    return subcat_models


@st.cache_resource
def train_part_terminology_models(training_data):
    part_term_models = {}

    for subcategory in training_data['Subcategory'].dropna().unique():
        subcat_data = training_data[training_data['Subcategory'] == subcategory]
        X_part = subcat_data['Product Title'].fillna('').astype(str)
        y_part = subcat_data['Part Terminology ID - Name'].astype(str)

        if y_part.nunique() > 1:
            try:
                pipeline = Pipeline([
                    ('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=5000)),
                    ('svm', LinearSVC(C=1.0, max_iter=2000, random_state=42))
                ])
                pipeline.fit(X_part, y_part)
                part_term_models[subcategory] = pipeline
            except Exception:
                st.warning(f"Could not train model for subcategory '{subcategory}'")
                part_term_models[subcategory] = y_part.iloc[0]
        else:
            part_term_models[subcategory] = y_part.iloc[0]

    return part_term_models


def get_top_predictions(model, text, top_n=3):
    """
    Return top N predicted classes safely from a LinearSVC pipeline.
    """
    try:
        scores = model.decision_function([text])

        # Binary classification can return shape (1,) instead of (1, n_classes)
        if np.ndim(scores) == 1:
            if len(model.classes_) == 2:
                # For binary, convert single score into two comparable scores
                score = float(scores[0])
                class_scores = np.array([-score, score])
            else:
                class_scores = np.array(scores)
        else:
            class_scores = np.array(scores[0])

        top_indices = np.argsort(class_scores)[-top_n:][::-1]
        preds = [model.classes_[i] for i in top_indices]
        while len(preds) < top_n:
            preds.append('N/A')
        return preds[:top_n]
    except Exception:
        return ['Error', 'N/A', 'N/A']


def hierarchical_prediction(item_description, category_pipeline, subcat_models, part_term_models):
    results = {
        'category': None,
        'subcategories': [],
        'part_terms': [],
        'error': None
    }

    try:
        item_description = '' if pd.isna(item_description) else str(item_description).strip()

        if not item_description:
            return ('Blank Description', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A')

        # Step 1: Category
        predicted_category = category_pipeline.predict([item_description])[0]
        results['category'] = predicted_category

        # Step 2: Subcategories
        if predicted_category in subcat_models:
            model = subcat_models[predicted_category]
            if isinstance(model, str):
                results['subcategories'] = [model, 'N/A', 'N/A']
            else:
                results['subcategories'] = get_top_predictions(model, item_description, top_n=3)
        else:
            results['subcategories'] = ['Unknown', 'N/A', 'N/A']

        # Step 3: Part Terminologies
        predicted_subcat = results['subcategories'][0]
        if predicted_subcat in part_term_models:
            model = part_term_models[predicted_subcat]
            if isinstance(model, str):
                results['part_terms'] = [model, 'N/A', 'N/A']
            else:
                results['part_terms'] = get_top_predictions(model, item_description, top_n=3)
        else:
            results['part_terms'] = ['Unknown', 'N/A', 'N/A']

    except Exception as e:
        results['error'] = str(e)
        results['category'] = 'Error'
        results['subcategories'] = ['Error', 'N/A', 'N/A']
        results['part_terms'] = ['Error', 'N/A', 'N/A']

    return (
        results['category'],
        results['subcategories'][0],
        results['subcategories'][1],
        results['subcategories'][2],
        results['part_terms'][0],
        results['part_terms'][1],
        results['part_terms'][2]
    )


def generate_template():
    template_df = pd.DataFrame({
        'Item Number': ['ITEM001', 'ITEM002', 'ITEM003'],
        'Description': [
            'Example: Heavy-duty stainless steel fastener',
            'Example: Industrial grade adhesive compound',
            'Example: Precision measurement tool'
        ]
    })
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        template_df.to_excel(writer, index=False, sheet_name='Items')
        worksheet = writer.sheets['Items']
        worksheet.set_column('A:A', 15)
        worksheet.set_column('B:B', 40)
    buffer.seek(0)
    return buffer


def display_results_table(df, mode='manual'):
    st.markdown("<div class='table-container'>", unsafe_allow_html=True)

    display_df = df.copy()

    if mode == 'manual' and 'Item' in display_df.columns:
        preferred_cols = [
            'Item',
            'Predicted Category',
            'Predicted Subcategory 1',
            'Predicted Subcategory 2',
            'Predicted Subcategory 3',
            'Predicted Part Terminology 1',
            'Predicted Part Terminology 2',
            'Predicted Part Terminology 3'
        ]
        existing_cols = [c for c in preferred_cols if c in display_df.columns]
        display_df = display_df[existing_cols]

    if mode == 'batch':
        preferred_cols = [
            'Item Number',
            'Description',
            'Predicted Category',
            'Predicted Subcategory 1',
            'Predicted Subcategory 2',
            'Predicted Subcategory 3',
            'Predicted Part Terminology 1',
            'Predicted Part Terminology 2',
            'Predicted Part Terminology 3'
        ]
        existing_cols = [c for c in preferred_cols if c in display_df.columns]
        display_df = display_df[existing_cols]

    st.dataframe(display_df, use_container_width=True, height=400)
    st.markdown("</div>", unsafe_allow_html=True)


def export_results(df, format_type='csv'):
    if format_type == 'csv':
        return df.to_csv(index=False).encode('utf-8')
    elif format_type == 'excel':
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Predictions')
        buffer.seek(0)
        return buffer.getvalue()


# ==================== MAIN APP ====================
col1, col2 = st.columns([3, 1])
with col1:
    st.title("📦 Hierarchical Item Categorization")
    st.markdown("*Machine Learning-Powered Product Classification System*")
with col2:
    st.markdown("""
    <div class='metric-card' style='text-align: center;'>
        <div style='font-size: 12px; color: #666;'>Status</div>
        <div style='font-size: 20px; font-weight: bold; color: #00a86b;'>✓ Ready</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class='info-box'>
💡 <strong>How it works:</strong> This system uses machine learning to automatically categorize items into subcategories and part terminology based on product descriptions. Choose between manual entry or batch file upload.
</div>
""", unsafe_allow_html=True)

# ==================== DATA LOADING & MODEL TRAINING ====================
with st.spinner("🔄 Loading training data and initializing models..."):
    training_data = load_data(CSV_URL)

    missing_cols = [col for col in REQUIRED_COLUMNS if col not in training_data.columns]
    if missing_cols:
        st.error(f"❌ Training data missing columns: {', '.join(missing_cols)}")
        st.stop()

    training_data['Product Title'] = training_data['Product Title'].fillna('').astype(str)
    training_data['Category'] = training_data['Category'].astype(str)
    training_data['Subcategory'] = training_data['Subcategory'].astype(str)
    training_data['Part Terminology ID - Name'] = training_data['Part Terminology ID - Name'].astype(str)

    training_data = training_data.dropna(subset=['Category', 'Subcategory', 'Part Terminology ID - Name'])

    category_pipeline = train_category_model(training_data['Product Title'], training_data['Category'])
    subcat_models = train_subcategory_models(training_data)
    part_term_models = train_part_terminology_models(training_data)

    st.session_state.training_complete = True

# Data stats
with st.expander("📊 Training Data Statistics", expanded=False):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <div style='color: #666; font-size: 12px;'>Total Items</div>
            <div style='font-size: 28px; font-weight: bold;'>{len(training_data)}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class='metric-card'>
            <div style='color: #666; font-size: 12px;'>Categories</div>
            <div style='font-size: 28px; font-weight: bold;'>{training_data['Category'].nunique()}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class='metric-card'>
            <div style='color: #666; font-size: 12px;'>Subcategories</div>
            <div style='font-size: 28px; font-weight: bold;'>{training_data['Subcategory'].nunique()}</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class='metric-card'>
            <div style='color: #666; font-size: 12px;'>Part Types</div>
            <div style='font-size: 28px; font-weight: bold;'>{training_data['Part Terminology ID - Name'].nunique()}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# ==================== TAB INTERFACE ====================
tab1, tab2, tab3 = st.tabs(["✍️ Manual Entry", "📤 Batch Upload", "📋 Template & Help"])

# ==================== TAB 1: MANUAL ENTRY ====================
with tab1:
    st.header("Enter Item Descriptions")
    st.markdown("Type or paste item descriptions below (one per line) to get instant categorization predictions.")

    item_input = st.text_area(
        "Item Descriptions:",
        height=180,
        placeholder="Example:\nStainless steel fastener M8x20\nIndustrial-grade silicone adhesive\nPrecision measurement gauge"
    )

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        predict_manual = st.button("🔍 Predict Categories", key='predict_manual', use_container_width=True)

    if predict_manual:
        items = [item.strip() for item in item_input.split("\n") if item.strip()]

        if not items:
            st.markdown("""
            <div class='warning-box'>
            ⚠️ <strong>No items entered:</strong> Please enter at least one item description.
            </div>
            """, unsafe_allow_html=True)
        else:
            if len(items) > MAX_BATCH_SIZE:
                st.markdown(f"""
                <div class='warning-box'>
                ⚠️ <strong>Too many items:</strong> Maximum {MAX_BATCH_SIZE} items per batch. Processing first {MAX_BATCH_SIZE}.
                </div>
                """, unsafe_allow_html=True)
                items = items[:MAX_BATCH_SIZE]

            with st.spinner(f"⏳ Processing {len(items)} item(s)..."):
                try:
                    df_manual = pd.DataFrame({
                        'Item': items,
                        'Description Length': [len(str(item)) for item in items]
                    })

                    predictions = df_manual['Item'].apply(
                        lambda x: pd.Series(
                            hierarchical_prediction(x, category_pipeline, subcat_models, part_term_models)
                        )
                    )

                    df_manual[
                        [
                            'Predicted Category',
                            'Predicted Subcategory 1',
                            'Predicted Subcategory 2',
                            'Predicted Subcategory 3',
                            'Predicted Part Terminology 1',
                            'Predicted Part Terminology 2',
                            'Predicted Part Terminology 3'
                        ]
                    ] = predictions

                    st.session_state.last_predictions = df_manual

                    st.markdown("""
                    <div class='success-box'>
                    ✓ <strong>Predictions Complete!</strong> Results shown below.
                    </div>
                    """, unsafe_allow_html=True)

                    display_results_table(df_manual, mode='manual')

                    col1, col2 = st.columns(2)
                    with col1:
                        csv_data = export_results(df_manual, format_type='csv')
                        st.download_button(
                            "📥 Download as CSV",
                            data=csv_data,
                            file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    with col2:
                        excel_data = export_results(df_manual, format_type='excel')
                        st.download_button(
                            "📥 Download as Excel",
                            data=excel_data,
                            file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True
                        )

                except Exception as e:
                    st.markdown(f"""
                    <div class='warning-box'>
                    ❌ <strong>Error:</strong> {str(e)}
                    </div>
                    """, unsafe_allow_html=True)

# ==================== TAB 2: BATCH UPLOAD ====================
with tab2:
    st.header("Upload Excel File for Batch Processing")

    st.markdown("""
    Upload an Excel file with your items. The file must contain:
    - **Item Number** column
    - **Description** column
    """)

    uploaded_file = st.file_uploader(
        "📤 Select Excel File",
        type=["xlsx", "xls"],
        help="Upload an .xlsx or .xls file"
    )

    if uploaded_file is not None:
        try:
            input_data = pd.read_excel(uploaded_file)

            required_cols = ['Item Number', 'Description']
            missing_cols = [col for col in required_cols if col not in input_data.columns]

            if missing_cols:
                st.markdown(f"""
                <div class='warning-box'>
                ⚠️ <strong>Missing columns:</strong> {', '.join(missing_cols)}<br>
                Your file must contain 'Item Number' and 'Description' columns.
                </div>
                """, unsafe_allow_html=True)
            else:
                with st.expander("👀 Preview Uploaded Data", expanded=True):
                    st.dataframe(input_data.head(10), use_container_width=True)
                    st.markdown(f"**Total rows:** {len(input_data)}")

                if st.button("🚀 Process & Predict", use_container_width=True, key='predict_batch'):
                    if len(input_data) > MAX_BATCH_SIZE:
                        st.markdown(f"""
                        <div class='warning-box'>
                        ⚠️ <strong>Large file:</strong> Processing first {MAX_BATCH_SIZE} items.
                        </div>
                        """, unsafe_allow_html=True)
                        input_data = input_data.head(MAX_BATCH_SIZE).copy()

                    input_data['Description'] = input_data['Description'].fillna('').astype(str)

                    with st.spinner(f"⏳ Processing {len(input_data)} item(s)..."):
                        try:
                            predictions = input_data['Description'].apply(
                                lambda x: pd.Series(
                                    hierarchical_prediction(x, category_pipeline, subcat_models, part_term_models)
                                )
                            )

                            input_data[
                                [
                                    'Predicted Category',
                                    'Predicted Subcategory 1',
                                    'Predicted Subcategory 2',
                                    'Predicted Subcategory 3',
                                    'Predicted Part Terminology 1',
                                    'Predicted Part Terminology 2',
                                    'Predicted Part Terminology 3'
                                ]
                            ] = predictions

                            st.session_state.last_predictions = input_data

                            st.markdown("""
                            <div class='success-box'>
                            ✓ <strong>Batch Processing Complete!</strong>
                            </div>
                            """, unsafe_allow_html=True)

                            display_results_table(input_data, mode='batch')

                            col1, col2 = st.columns(2)
                            with col1:
                                csv_data = export_results(input_data, format_type='csv')
                                st.download_button(
                                    "📥 Download as CSV",
                                    data=csv_data,
                                    file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv",
                                    use_container_width=True
                                )
                            with col2:
                                excel_data = export_results(input_data, format_type='excel')
                                st.download_button(
                                    "📥 Download as Excel",
                                    data=excel_data,
                                    file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    use_container_width=True
                                )

                        except Exception as e:
                            st.markdown(f"""
                            <div class='warning-box'>
                            ❌ <strong>Error processing file:</strong> {str(e)}
                            </div>
                            """, unsafe_allow_html=True)

        except Exception as e:
            st.markdown(f"""
            <div class='warning-box'>
            ❌ <strong>Error reading file:</strong> {str(e)}<br>
            Make sure the file is a valid Excel file (.xlsx or .xls).
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='info-box'>
        ℹ️ <strong>No file selected yet.</strong> Upload an Excel file to get started. Need a template? Go to the "Template & Help" tab.
        </div>
        """, unsafe_allow_html=True)

# ==================== TAB 3: TEMPLATE & HELP ====================
with tab3:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📋 Download Template")
        st.markdown("""
        Use this template to prepare your batch file:
        - **Item Number**: Unique identifier
        - **Description**: Detailed product description
        """)

        st.download_button(
            label="📥 Download Excel Template",
            data=generate_template(),
            file_name="categorization_template.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

    with col2:
        st.subheader("🎯 Best Practices")
        st.markdown("""
        **For best results:**
        - Use detailed product descriptions
        - Include relevant specifications
        - Avoid very short descriptions
        - Use consistent terminology
        - Include units and measurements when relevant
        """)

    st.markdown("---")

    st.subheader("❓ Frequently Asked Questions")

    faq = {
        "What predictions do I get?": "The system provides a primary category, three subcategory predictions, and three part terminology predictions for each item.",
        "How accurate are predictions?": "Accuracy depends on training data quality and description clarity. Detailed descriptions typically yield better results.",
        "Can I reuse previous results?": "Yes. Download results as CSV/Excel and reuse them later.",
        "What's the batch limit?": f"Maximum {MAX_BATCH_SIZE} items per batch.",
        "How long does processing take?": "Processing speed depends on batch size and model size.",
        "Which file formats are supported?": "Both .xlsx and .xls Excel formats are supported."
    }

    for question, answer in faq.items():
        with st.expander(question):
            st.markdown(answer)

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 12px; padding: 20px;'>
    <strong>Hierarchical Item Categorization System</strong> |
    Built with Streamlit & Machine Learning
</div>
""", unsafe_allow_html=True)
