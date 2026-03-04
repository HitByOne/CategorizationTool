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
    """Create diverse training data for better predictions"""
    data = {
        'Product Title': [
            # Fasteners (20 items)
            'Stainless Steel Bolt M8x20', 'Steel Bolt M10x25', 'Brass Bolt M6x15',
            'Hex Nut M8 Stainless', 'Hex Nut M10 Steel', 'Washer Steel M8',
            'Phillips Head Screw 3mm', 'Flathead Screw 4mm', 'Anchor Bolt M12',
            'Toggle Bolt M6', 'U-Bolt Stainless M10', 'Eye Bolt M8',
            'Carriage Bolt M10', 'Machine Screw M5', 'Wood Screw 3 inch', 'Rivet Aluminium',
            'Cone Head Screw', 'T-Nut M8', 'Spring Washer', 'Lock Nut M6',
            
            # Adhesives (20 items)
            'Industrial Epoxy Resin 500ml', 'Two Part Epoxy Glue', 'Cyanoacrylate Super Glue',
            'Hot Melt Adhesive Sticks', 'Contact Cement Spray', 'Polyurethane Adhesive',
            'Silicone Sealant White', 'Acrylic Latex Caulk', 'Rubber Cement', 'Waterproof Adhesive',
            'Double Sided Tape Heavy Duty', 'Flexible Adhesive Sealant', 'Moisture Cure Polyurethane',
            'Structural Adhesive Paste', 'Peel and Stick Adhesive', 'Foam Tape Double Sided',
            'Pressure Sensitive Adhesive', 'Vinyl Adhesive', 'Elastic Adhesive', 'Synthetic Adhesive',
            
            # Tools (20 items)
            'Digital Caliper 150mm', 'Vernier Caliper Stainless', 'Micrometer 0-25mm',
            'Measuring Tape 10m', 'Analog Scale 0-1000g', 'Digital Scale 5kg',
            'Laser Distance Meter', 'Spirit Level 2m', 'Angle Finder Digital', 'Depth Gauge',
            'Precision Ruler 300mm', 'Thickness Gauge Digital', 'Pressure Gauge 0-10bar',
            'Temperature Gun Infrared', 'Multimeter Digital', 'Oscilloscope Probe',
            'Compass Precision', 'Protractor Metal', 'Straightedge Aluminum', 'Feeler Gauge Set',
            
            # Seals (20 items)
            'Rubber Gasket Nitrile NBR', 'Silicone Gasket Food Grade', 'EPDM Rubber Gasket',
            'Viton Gasket High Temp', 'Cork Gasket', 'Graphite Gasket',
            'Oil Seal 30x55x10', 'Mechanical Seal Assembly', 'Spring Seal Unit',
            'Bellows Seal', 'Labyrinth Seal', 'Packing Ring PTFE',
            'O-Ring Rubber 10mm', 'X-Ring Seal', 'Spiral Wound Gasket', 'Metal Ring Gasket',
            'Bonded Seal Washer', 'Compression Seal', 'Lip Seal Assembly', 'Face Seal',
            
            # Electrical (20 items)
            'Copper Wire 2mm Diameter', 'Aluminium Wire 1.5mm', 'Stranded Cable 4mm',
            'Twisted Pair Cable 10m', 'Coaxial Cable RG-58', 'Shielded Cable',
            'Fiber Optic Cable', 'USB Extension Cable 5m', 'HDMI Cable 2m', 'Power Cord 3m',
            'Antenna Cable', 'Network Ethernet Cable', 'Microphone Cable 5m', 'Speaker Wire 2.5mm',
            'Telephone Cable RJ-11', 'Control Cable Multi-conductor',
            'Data Cable Ribbon', 'Power Cable Heavy Duty', 'Audio Snake Cable', 'RF Cable',
            
            # Plastics (20 items)
            'PVC Pipe 50mm x 1m', 'PVC Tube 32mm Clear', 'Plastic Sleeve Shrink',
            'Polycarbonate Sheet 5mm', 'Acrylic Sheet Transparent', 'HDPE Film Roll',
            'Rubber Hose 10mm', 'Silicone Hose 8mm', 'Vinyl Tubing Clear 6mm',
            'Plastic Bushing 10mm', 'Polymer Bearing', 'Nylon Spacer Ring',
            'Delrin Rod 10mm', 'Teflon Washer', 'Plastic Connector 3-way', 'Rubber Damper Block',
            'PVC Fitting Elbow', 'Plastic Clamp', 'Nylon Collar', 'Polyurethane Wheel',
        ],
        'Category': [
            # Fasteners (20)
            'Fasteners', 'Fasteners', 'Fasteners',
            'Fasteners', 'Fasteners', 'Fasteners',
            'Fasteners', 'Fasteners', 'Fasteners',
            'Fasteners', 'Fasteners', 'Fasteners',
            'Fasteners', 'Fasteners', 'Fasteners', 'Fasteners',
            'Fasteners', 'Fasteners', 'Fasteners', 'Fasteners',
            
            # Adhesives (20)
            'Adhesives', 'Adhesives', 'Adhesives',
            'Adhesives', 'Adhesives', 'Adhesives',
            'Adhesives', 'Adhesives', 'Adhesives', 'Adhesives',
            'Adhesives', 'Adhesives', 'Adhesives',
            'Adhesives', 'Adhesives', 'Adhesives',
            'Adhesives', 'Adhesives', 'Adhesives', 'Adhesives',
            
            # Tools (20)
            'Tools', 'Tools', 'Tools',
            'Tools', 'Tools', 'Tools',
            'Tools', 'Tools', 'Tools', 'Tools',
            'Tools', 'Tools', 'Tools',
            'Tools', 'Tools', 'Tools',
            'Tools', 'Tools', 'Tools', 'Tools',
            
            # Seals (20)
            'Seals', 'Seals', 'Seals',
            'Seals', 'Seals', 'Seals',
            'Seals', 'Seals', 'Seals',
            'Seals', 'Seals', 'Seals',
            'Seals', 'Seals', 'Seals', 'Seals',
            'Seals', 'Seals', 'Seals', 'Seals',
            
            # Electrical (20)
            'Electrical', 'Electrical', 'Electrical',
            'Electrical', 'Electrical', 'Electrical',
            'Electrical', 'Electrical', 'Electrical', 'Electrical',
            'Electrical', 'Electrical', 'Electrical',
            'Electrical', 'Electrical', 'Electrical',
            'Electrical', 'Electrical', 'Electrical', 'Electrical',
            
            # Plastics (20)
            'Plastics', 'Plastics', 'Plastics',
            'Plastics', 'Plastics', 'Plastics',
            'Plastics', 'Plastics', 'Plastics',
            'Plastics', 'Plastics', 'Plastics',
            'Plastics', 'Plastics', 'Plastics', 'Plastics',
            'Plastics', 'Plastics', 'Plastics', 'Plastics',
        ],
        'Subcategory': [
            # Fasteners (20)
            'Bolts', 'Bolts', 'Bolts',
            'Nuts', 'Nuts', 'Washers',
            'Screws', 'Screws', 'Anchors',
            'Anchors', 'Bolts', 'Bolts',
            'Bolts', 'Screws', 'Screws', 'Rivets',
            'Screws', 'Nuts', 'Washers', 'Nuts',
            
            # Adhesives (20)
            'Epoxy', 'Epoxy', 'Cyanoacrylate',
            'Hot Melt', 'Spray', 'Polyurethane',
            'Silicone', 'Caulk', 'Rubber Cement', 'Waterproof',
            'Tape', 'Sealant', 'Moisture Cure',
            'Structural', 'Peel Stick', 'Foam Tape',
            'Pressure Sensitive', 'Vinyl', 'Elastic', 'Synthetic',
            
            # Tools (20)
            'Calipers', 'Calipers', 'Micrometers',
            'Measuring', 'Scales', 'Digital Scales',
            'Laser', 'Levels', 'Angle Finder', 'Gauges',
            'Rulers', 'Digital Gauges', 'Pressure',
            'Infrared', 'Multimeter', 'Oscilloscope',
            'Navigation', 'Angle', 'Straightedges', 'Feeler',
            
            # Seals (20)
            'Gaskets', 'Gaskets', 'Gaskets',
            'Gaskets', 'Gaskets', 'Gaskets',
            'Oil Seals', 'Mechanical Seals', 'Spring Seals',
            'Bellows', 'Labyrinth', 'Packing',
            'O-Rings', 'X-Rings', 'Spiral', 'Metal Rings',
            'Bonded', 'Compression', 'Lip', 'Face',
            
            # Electrical (20)
            'Wire', 'Wire', 'Cable',
            'Cable', 'Cable', 'Cable',
            'Fiber', 'Cable', 'Cable', 'Power',
            'Antenna', 'Network', 'Audio',
            'Audio', 'Telecom', 'Control',
            'Data', 'Power', 'Audio', 'RF',
            
            # Plastics (20)
            'Pipe', 'Tube', 'Shrink',
            'Sheet', 'Sheet', 'Film',
            'Hose', 'Hose', 'Tubing',
            'Bushings', 'Bearings', 'Spacers',
            'Rod', 'Washer', 'Connectors', 'Dampers',
            'Fittings', 'Clamps', 'Collars', 'Wheels',
        ]
    }
    return pd.DataFrame(data)

# ==================== TRAIN MODELS ====================
@st.cache_resource
def train_models(data):
    """Train improved category and subcategory models"""
    # Category model with better parameters
    cat_model = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=200,           # More features for better differentiation
            ngram_range=(1, 3),         # Include trigrams
            min_df=1,                   # Include rare words
            sublinear_tf=True           # Use sublinear term frequency scaling
        )),
        ('svm', LinearSVC(
            max_iter=2000,              # More iterations for better convergence
            random_state=42,
            C=0.5,                      # Lower C for more regularization (softer margin)
            class_weight='balanced'     # Handle imbalanced classes
        ))
    ])
    cat_model.fit(data['Product Title'].values, data['Category'].values)
    
    # Subcategory models (one per category) with better parameters
    sub_models = {}
    for category in data['Category'].unique():
        cat_data = data[data['Category'] == category]
        n_subcats = len(cat_data['Subcategory'].unique())
        
        if n_subcats > 1:
            sub_model = Pipeline([
                ('tfidf', TfidfVectorizer(
                    max_features=150,
                    ngram_range=(1, 2),
                    min_df=1,
                    sublinear_tf=True
                )),
                ('svm', LinearSVC(
                    max_iter=2000,
                    random_state=42,
                    C=0.5,
                    class_weight='balanced'
                ))
            ])
            sub_model.fit(cat_data['Product Title'].values, cat_data['Subcategory'].values)
            sub_models[category] = sub_model
        else:
            # Only one subcategory, just store it
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
