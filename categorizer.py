import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
import io

# ==================== CONFIGURATION ====================
st.set_page_config(
    page_title="Item Categorization",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==================== STYLE ====================
st.markdown("""
<style>
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
.main { padding: 2rem; }
h1 { color: #0066cc; margin-bottom: 0.5rem; }
.stTabs [data-baseweb="tab-list"] button { font-size: 16px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ==================== TITLE ====================
st.title("📦 Item Categorization System")
st.markdown("Simple machine learning-based product categorizer")

# ==================== SAMPLE DATA ====================
@st.cache_data
def get_sample_data():
    """Create sample training data for demonstration"""
    data = {
        'Product Title': [
            'Stainless Steel Fastener M8x20', 'Stainless Steel Bolt 10mm',
            'Industrial Adhesive Epoxy', 'Industrial Glue Type A',
            'Digital Caliper 150mm', 'Digital Ruler Precision',
            'Rubber Gasket A', 'Rubber Seal Type B',
            'Copper Wire 2mm', 'Copper Cable Strand',
            'Plastic Sleeve 50mm', 'Plastic Tube PVC',
        ],
        'Category': [
            'Fasteners', 'Fasteners', 
            'Adhesives', 'Adhesives',
            'Tools', 'Tools',
            'Seals', 'Seals',
            'Electrical', 'Electrical',
            'Plastics', 'Plastics'
        ],
        'Subcategory': [
            'Bolts', 'Bolts',
            'Epoxy', 'Epoxy',
            'Measuring', 'Measuring',
            'Gaskets', 'Gaskets',
            'Conductors', 'Conductors',
            'Tubes', 'Tubes'
        ]
    }
    return pd.DataFrame(data)

# ==================== TRAIN MODELS ====================
@st.cache_resource
def train_models(data):
    """Train simple category and subcategory models"""
    # Category model
    cat_model = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=100, ngram_range=(1, 2))),
        ('svm', LinearSVC(max_iter=1000, random_state=42))
    ])
    cat_model.fit(data['Product Title'], data['Category'])
    
    # Subcategory models (one per category)
    sub_models = {}
    for category in data['Category'].unique():
        cat_data = data[data['Category'] == category]
        if len(cat_data['Subcategory'].unique()) > 1:
            sub_model = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=100, ngram_range=(1, 2))),
                ('svm', LinearSVC(max_iter=1000, random_state=42))
            ])
            sub_model.fit(cat_data['Product Title'], cat_data['Subcategory'])
            sub_models[category] = sub_model
        else:
            sub_models[category] = cat_data['Subcategory'].iloc[0]
    
    return cat_model, sub_models

# Load data and train
sample_data = get_sample_data()
cat_model, sub_models = train_models(sample_data)

# ==================== PREDICTION FUNCTION ====================
def predict_item(description):
    """Predict category and subcategory for an item"""
    try:
        # Predict category
        category = cat_model.predict([description])[0]
        
        # Predict subcategory
        if category in sub_models:
            if isinstance(sub_models[category], str):
                subcategory = sub_models[category]
            else:
                subcategory = sub_models[category].predict([description])[0]
        else:
            subcategory = "Unknown"
        
        return category, subcategory, "✓ Success"
    except Exception as e:
        return "Error", "Error", f"✗ {str(e)}"

# ==================== USER INTERFACE ====================
tab1, tab2, tab3 = st.tabs(["📝 Single Item", "📤 Batch Upload", "ℹ️ Info"])

# ==================== TAB 1: SINGLE ITEM ====================
with tab1:
    st.subheader("Enter One Item")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        item_input = st.text_input(
            "Product Description:",
            placeholder="e.g., Stainless steel fastener M8x20",
            label_visibility="collapsed"
        )
    
    with col2:
        predict_btn = st.button("🔍 Predict", use_container_width=True)
    
    if predict_btn and item_input:
        with st.spinner("Processing..."):
            category, subcategory, status = predict_item(item_input)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Category", category)
            with col2:
                st.metric("Subcategory", subcategory)
            with col3:
                st.metric("Status", status)
    
    elif predict_btn:
        st.warning("⚠️ Please enter a product description")

# ==================== TAB 2: BATCH UPLOAD ====================
with tab2:
    st.subheader("Upload Excel File")
    st.write("Upload a file with columns: **Item Number** and **Description**")
    
    uploaded_file = st.file_uploader("Choose Excel file", type=["xlsx", "xls"])
    
    if uploaded_file:
        try:
            df = pd.read_excel(uploaded_file)
            
            if 'Item Number' not in df.columns or 'Description' not in df.columns:
                st.error("❌ File must have 'Item Number' and 'Description' columns")
            else:
                st.write(f"**Loaded {len(df)} items**")
                
                if st.button("🚀 Process All", use_container_width=True):
                    with st.spinner("Processing..."):
                        results = []
                        for idx, row in df.iterrows():
                            cat, subcat, _ = predict_item(row['Description'])
                            results.append({
                                'Item Number': row['Item Number'],
                                'Description': row['Description'],
                                'Category': cat,
                                'Subcategory': subcat
                            })
                        
                        results_df = pd.DataFrame(results)
                        st.success("✓ Complete!")
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Download button
                        csv = results_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "📥 Download CSV",
                            csv,
                            "results.csv",
                            "text/csv",
                            use_container_width=True
                        )
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# ==================== TAB 3: INFO ====================
with tab3:
    st.subheader("About This App")
    
    st.markdown("""
    ### How It Works
    This app uses machine learning to automatically categorize products based on their descriptions.
    
    ### Features
    - ✅ Single item prediction
    - ✅ Batch file processing
    - ✅ CSV export
    - ✅ Simple and fast
    
    ### Sample Categories
    The demo includes: Fasteners, Adhesives, Tools, Seals, Electrical, Plastics
    
    ### How to Use
    1. **Single Item**: Enter a product description and click "Predict"
    2. **Batch**: Upload an Excel file with Item Number and Description columns
    3. **Download**: Export results as CSV
    
    ### Technical Details
    - **ML Algorithm**: Support Vector Machine (SVM)
    - **Text Processing**: TF-IDF Vectorization
    - **Framework**: Streamlit + scikit-learn
    """)
    
    st.divider()
    
    st.markdown("""
    ### Sample Training Data
    """)
    st.dataframe(sample_data, use_container_width=True)

# ==================== FOOTER ====================
st.divider()
st.markdown("""
<div style='text-align: center; color: #999; font-size: 12px; padding: 20px;'>
Item Categorization System | Powered by Streamlit & scikit-learn
</div>
""", unsafe_allow_html=True)
