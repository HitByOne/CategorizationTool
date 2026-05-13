import io
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")

# ==================== PAGE CONFIG ====================
st.set_page_config(
    layout="wide",
    page_title="Item Categorization",
    initial_sidebar_state="collapsed",
    menu_items={
        "About": "### Hierarchical Item Categorization System\nA machine learning-powered tool for intelligent product categorization."
    }
)

# ==================== CUSTOM STYLING ====================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

    :root {
        --bg: #0f1117;
        --surface: #1a1d27;
        --surface2: #22263a;
        --border: #2e334d;
        --accent: #6c63ff;
        --accent2: #00d2c8;
        --accent3: #ff6b6b;
        --text: #e8eaf6;
        --text-muted: #8890b0;
        --success: #00d2a0;
        --warning: #ffb347;
        --danger: #ff6b6b;
        --radius: 12px;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        background-color: var(--bg) !important;
        color: var(--text) !important;
    }

    /* Global dark overrides */
    .stApp {
        background-color: var(--bg) !important;
    }

    .block-container {
        padding-top: 2rem;
        max-width: 1400px;
    }

    /* Header */
    .app-header {
        background: linear-gradient(135deg, #1a1d27 0%, #22263a 100%);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 32px 36px;
        margin-bottom: 28px;
        position: relative;
        overflow: hidden;
    }
    .app-header::before {
        content: '';
        position: absolute;
        top: -40px; right: -40px;
        width: 200px; height: 200px;
        background: radial-gradient(circle, rgba(108,99,255,0.15) 0%, transparent 70%);
        border-radius: 50%;
    }
    .app-header h1 {
        font-family: 'Space Mono', monospace;
        font-size: 28px;
        font-weight: 700;
        color: var(--text) !important;
        margin: 0 0 8px 0;
        letter-spacing: -0.5px;
    }
    .app-header p {
        color: var(--text-muted) !important;
        font-size: 15px;
        margin: 0;
    }
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(0, 210, 160, 0.12);
        border: 1px solid rgba(0, 210, 160, 0.3);
        color: var(--success) !important;
        padding: 6px 14px;
        border-radius: 100px;
        font-size: 13px;
        font-weight: 600;
        margin-top: 16px;
    }

    /* Metric cards */
    .metric-row { display: flex; gap: 16px; margin-bottom: 28px; flex-wrap: wrap; }
    .metric-card {
        flex: 1; min-width: 160px;
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 20px 24px;
        position: relative;
        overflow: hidden;
    }
    .metric-card::after {
        content: '';
        position: absolute;
        bottom: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, var(--accent), var(--accent2));
    }
    .metric-card .label {
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: var(--text-muted) !important;
        margin-bottom: 8px;
    }
    .metric-card .value {
        font-family: 'Space Mono', monospace;
        font-size: 32px;
        font-weight: 700;
        color: var(--text) !important;
        line-height: 1;
    }

    /* Info / alert boxes */
    .alert {
        padding: 14px 18px;
        border-radius: var(--radius);
        margin: 12px 0;
        font-size: 14px;
        border-left: 3px solid;
    }
    .alert-info {
        background: rgba(108,99,255,0.08);
        border-color: var(--accent);
        color: #c7c4ff !important;
    }
    .alert-success {
        background: rgba(0,210,160,0.08);
        border-color: var(--success);
        color: #a0f5e5 !important;
    }
    .alert-warning {
        background: rgba(255,179,71,0.08);
        border-color: var(--warning);
        color: #ffe0a8 !important;
    }
    .alert-danger {
        background: rgba(255,107,107,0.08);
        border-color: var(--danger);
        color: #ffc5c5 !important;
    }

    /* Confidence badge */
    .conf-high   { color: #00d2a0 !important; font-weight: 600; }
    .conf-medium { color: #ffb347 !important; font-weight: 600; }
    .conf-low    { color: #ff6b6b !important; font-weight: 600; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--surface) !important;
        border-radius: var(--radius) var(--radius) 0 0;
        padding: 8px 8px 0;
        gap: 4px;
        border-bottom: 1px solid var(--border);
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        border-radius: 8px 8px 0 0 !important;
        color: var(--text-muted) !important;
        font-weight: 500;
        font-size: 14px;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background: var(--surface2) !important;
        color: var(--text) !important;
    }
    .stTabs [data-baseweb="tab-panel"] {
        background: var(--surface);
        border: 1px solid var(--border);
        border-top: none;
        border-radius: 0 0 var(--radius) var(--radius);
        padding: 24px;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, var(--accent), #8b83ff) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        padding: 10px 24px !important;
        transition: opacity 0.2s, transform 0.1s !important;
    }
    .stButton > button:hover {
        opacity: 0.9 !important;
        transform: translateY(-1px) !important;
    }
    .stDownloadButton > button {
        background: var(--surface2) !important;
        color: var(--text) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
    }

    /* Text inputs */
    .stTextArea textarea, .stTextInput input {
        background: var(--surface2) !important;
        border: 1px solid var(--border) !important;
        color: var(--text) !important;
        border-radius: 8px !important;
        font-family: 'DM Sans', sans-serif !important;
    }

    /* File uploader */
    .stFileUploader {
        background: var(--surface2) !important;
        border: 1px dashed var(--border) !important;
        border-radius: var(--radius) !important;
    }

    /* Dataframe */
    .stDataFrame {
        border-radius: var(--radius) !important;
        overflow: hidden !important;
    }
    iframe[title="st_aggrid"] { border-radius: var(--radius); }

    /* Expander */
    .streamlit-expanderHeader {
        background: var(--surface2) !important;
        border-radius: 8px !important;
        color: var(--text) !important;
        font-weight: 500 !important;
    }

    /* Selectbox / multiselect */
    .stSelectbox > div, .stMultiSelect > div {
        background: var(--surface2) !important;
    }

    /* Progress bar */
    .stProgress > div > div { background: var(--accent) !important; }

    /* Spinner */
    .stSpinner > div { border-top-color: var(--accent) !important; }

    /* Section headers */
    .section-title {
        font-family: 'Space Mono', monospace;
        font-size: 16px;
        font-weight: 700;
        color: var(--text) !important;
        margin-bottom: 16px;
        padding-bottom: 10px;
        border-bottom: 1px solid var(--border);
    }

    /* Search bar container */
    .search-row {
        display: flex;
        gap: 12px;
        align-items: flex-end;
        margin-bottom: 16px;
    }

    /* Divider */
    hr { border-color: var(--border) !important; }

    /* Markdown text */
    .stMarkdown p, .stMarkdown li { color: var(--text-muted) !important; }
    .stMarkdown strong { color: var(--text) !important; }

    h1, h2, h3 { color: var(--text) !important; }
</style>
""", unsafe_allow_html=True)

# ==================== CONSTANTS ====================
CSV_URL = "https://drive.google.com/uc?id=1cnau3XSlOjG4m9RZyk5UXTVakwPfuori&export=download"
REQUIRED_COLUMNS = ['Product Title', 'Category', 'Subcategory', 'Part Terminology ID - Name']
MAX_BATCH_SIZE = 5000

# ==================== SESSION STATE ====================
for key, val in [
    ('training_complete', False),
    ('last_predictions', None),
    ('filter_text', ''),
    ('filter_category', 'All'),
]:
    if key not in st.session_state:
        st.session_state[key] = val

# ==================== DATA & MODEL FUNCTIONS ====================
@st.cache_data(show_spinner=False)
def load_data(url):
    try:
        data = pd.read_csv(url, on_bad_lines='skip')
        return data
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        st.stop()


def _build_pipeline():
    return Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=5000)),
        ('svm', LinearSVC(C=1.0, max_iter=2000, random_state=42))
    ])


@st.cache_resource(show_spinner=False)
def train_all_models(csv_url):
    """Train all models in parallel for faster startup."""
    data = load_data(csv_url)
    missing = [c for c in REQUIRED_COLUMNS if c not in data.columns]
    if missing:
        st.error(f"❌ Missing columns: {', '.join(missing)}")
        st.stop()

    data['Product Title'] = data['Product Title'].fillna('').astype(str)
    data['Category'] = data['Category'].astype(str)
    data['Subcategory'] = data['Subcategory'].astype(str)
    data['Part Terminology ID - Name'] = data['Part Terminology ID - Name'].astype(str)
    data = data.dropna(subset=['Category', 'Subcategory', 'Part Terminology ID - Name'])

    # --- Category model ---
    cat_pipeline = _build_pipeline()
    cat_pipeline.fit(data['Product Title'], data['Category'])

    categories = data['Category'].dropna().unique()
    subcategories = data['Subcategory'].dropna().unique()

    def train_subcat(category):
        cat_data = data[data['Category'] == category]
        X = cat_data['Product Title'].fillna('').astype(str)
        y = cat_data['Subcategory'].astype(str)
        if y.nunique() > 1:
            p = _build_pipeline()
            p.fit(X, y)
            return category, p
        return category, y.iloc[0]

    def train_partterm(subcategory):
        sub_data = data[data['Subcategory'] == subcategory]
        X = sub_data['Product Title'].fillna('').astype(str)
        y = sub_data['Part Terminology ID - Name'].astype(str)
        if y.nunique() > 1:
            p = _build_pipeline()
            p.fit(X, y)
            return subcategory, p
        return subcategory, y.iloc[0]

    with ThreadPoolExecutor() as executor:
        subcat_results = list(executor.map(train_subcat, categories))
        partterm_results = list(executor.map(train_partterm, subcategories))

    subcat_models = dict(subcat_results)
    part_term_models = dict(partterm_results)

    return data, cat_pipeline, subcat_models, part_term_models


def get_top_predictions_with_confidence(model, text, top_n=3):
    """Return top N (prediction, confidence_pct) tuples from a LinearSVC pipeline."""
    try:
        scores = model.decision_function([text])
        if np.ndim(scores) == 1:
            if len(model.classes_) == 2:
                score = float(scores[0])
                class_scores = np.array([-score, score])
            else:
                class_scores = np.array(scores)
        else:
            class_scores = np.array(scores[0])

        # Softmax-style normalization for confidence
        exp_scores = np.exp(class_scores - np.max(class_scores))
        confidences = exp_scores / exp_scores.sum()

        top_indices = np.argsort(class_scores)[-top_n:][::-1]
        results = []
        for i in top_indices:
            results.append((model.classes_[i], round(float(confidences[i]) * 100, 1)))
        while len(results) < top_n:
            results.append(('N/A', 0.0))
        return results[:top_n]
    except Exception:
        return [('Error', 0.0), ('N/A', 0.0), ('N/A', 0.0)]


def hierarchical_prediction(item_description, category_pipeline, subcat_models, part_term_models):
    try:
        item_description = '' if pd.isna(item_description) else str(item_description).strip()
        if not item_description:
            return {
                'category': 'Blank Description', 'category_conf': 0,
                'subcats': [('N/A', 0)] * 3,
                'partterms': [('N/A', 0)] * 3,
            }

        # Category
        predicted_category = category_pipeline.predict([item_description])[0]
        # Confidence for category
        cat_scores = category_pipeline.decision_function([item_description])
        if np.ndim(cat_scores) == 1:
            cat_exp = np.exp(cat_scores - np.max(cat_scores))
        else:
            cat_exp = np.exp(cat_scores[0] - np.max(cat_scores[0]))
        cat_conf = round(float(np.max(cat_exp / cat_exp.sum())) * 100, 1)

        # Subcategory
        if predicted_category in subcat_models:
            model = subcat_models[predicted_category]
            if isinstance(model, str):
                subcats = [(model, 100.0), ('N/A', 0.0), ('N/A', 0.0)]
            else:
                subcats = get_top_predictions_with_confidence(model, item_description, top_n=3)
        else:
            subcats = [('Unknown', 0.0), ('N/A', 0.0), ('N/A', 0.0)]

        # Part Terminology
        predicted_subcat = subcats[0][0]
        if predicted_subcat in part_term_models:
            model = part_term_models[predicted_subcat]
            if isinstance(model, str):
                partterms = [(model, 100.0), ('N/A', 0.0), ('N/A', 0.0)]
            else:
                partterms = get_top_predictions_with_confidence(model, item_description, top_n=3)
        else:
            partterms = [('Unknown', 0.0), ('N/A', 0.0), ('N/A', 0.0)]

        return {
            'category': predicted_category,
            'category_conf': cat_conf,
            'subcats': subcats,
            'partterms': partterms,
        }

    except Exception as e:
        return {
            'category': 'Error', 'category_conf': 0,
            'subcats': [('Error', 0.0)] * 3,
            'partterms': [('Error', 0.0)] * 3,
        }


def predictions_to_df(items, results_list, has_item_number=False, item_numbers=None):
    rows = []
    for i, (item, r) in enumerate(zip(items, results_list)):
        row = {}
        if has_item_number and item_numbers is not None:
            row['Item Number'] = item_numbers[i]
        row['Description'] = item
        row['Category'] = r['category']
        row['Category Confidence'] = f"{r['category_conf']}%"
        for j, (sub, conf) in enumerate(r['subcats'], 1):
            row[f'Subcategory {j}'] = sub
            row[f'Subcategory {j} Confidence'] = f"{conf}%" if sub not in ('N/A', 'Error') else ''
        for j, (pt, conf) in enumerate(r['partterms'], 1):
            row[f'Part Terminology {j}'] = pt
            row[f'Part Terminology {j} Confidence'] = f"{conf}%" if pt not in ('N/A', 'Error') else ''
        rows.append(row)
    return pd.DataFrame(rows)


def generate_template():
    template_df = pd.DataFrame({
        'Item Number': ['ITEM001', 'ITEM002', 'ITEM003'],
        'Description': [
            'Heavy-duty stainless steel fastener M8x20',
            'Industrial grade silicone adhesive compound',
            'Precision digital measurement gauge'
        ]
    })
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        template_df.to_excel(writer, index=False, sheet_name='Items')
        ws = writer.sheets['Items']
        ws.set_column('A:A', 15)
        ws.set_column('B:B', 45)
    buffer.seek(0)
    return buffer


def export_results(df, format_type='csv'):
    if format_type == 'csv':
        return df.to_csv(index=False).encode('utf-8')
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Predictions')
    buffer.seek(0)
    return buffer.getvalue()


def conf_color_class(conf_str):
    """Return CSS class based on confidence value."""
    try:
        val = float(conf_str.replace('%', ''))
        if val >= 70:
            return 'conf-high'
        elif val >= 40:
            return 'conf-medium'
        return 'conf-low'
    except Exception:
        return ''


def render_results_section(df, mode='manual'):
    """Render filterable, searchable results table + charts."""

    # --- Search & Filter bar ---
    col_s, col_f = st.columns([3, 2])
    with col_s:
        search = st.text_input("🔍 Search descriptions", placeholder="Type to filter rows…", key=f'search_{mode}')
    with col_f:
        cats = ['All'] + sorted(df['Category'].dropna().unique().tolist())
        cat_filter = st.selectbox("Filter by Category", cats, key=f'catfilter_{mode}')

    filtered = df.copy()
    if search:
        mask = filtered['Description'].str.contains(search, case=False, na=False)
        filtered = filtered[mask]
    if cat_filter != 'All':
        filtered = filtered[filtered['Category'] == cat_filter]

    st.markdown(f"<p style='color:var(--text-muted);font-size:13px;margin-bottom:8px;'>{len(filtered)} of {len(df)} items shown</p>", unsafe_allow_html=True)
    st.dataframe(filtered, use_container_width=True, height=380)

    # --- Charts ---
    st.markdown("<div class='section-title' style='margin-top:28px;'>📊 Prediction Insights</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        cat_counts = df['Category'].value_counts().reset_index()
        cat_counts.columns = ['Category', 'Count']
        fig = px.bar(
            cat_counts.head(12), x='Count', y='Category', orientation='h',
            title='Category Distribution',
            color='Count',
            color_continuous_scale=['#6c63ff', '#00d2c8'],
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#8890b0',
            title_font_color='#e8eaf6',
            coloraxis_showscale=False,
            margin=dict(l=0, r=0, t=40, b=0),
            yaxis=dict(tickfont=dict(size=11)),
        )
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        # Parse confidence values
        conf_col = 'Category Confidence'
        if conf_col in df.columns:
            confs = df[conf_col].str.replace('%', '').astype(float)
            buckets = pd.cut(confs, bins=[0, 40, 70, 100], labels=['Low (<40%)', 'Medium (40–70%)', 'High (>70%)'])
            bucket_counts = buckets.value_counts().reindex(['High (>70%)', 'Medium (40–70%)', 'Low (<40%)'])
            fig2 = go.Figure(go.Pie(
                labels=bucket_counts.index,
                values=bucket_counts.values,
                hole=0.55,
                marker_colors=['#00d2a0', '#ffb347', '#ff6b6b'],
                textfont_size=12,
            ))
            fig2.update_layout(
                title='Category Confidence Levels',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#8890b0',
                title_font_color='#e8eaf6',
                margin=dict(l=0, r=0, t=40, b=0),
                legend=dict(font=dict(color='#8890b0')),
                showlegend=True,
            )
            st.plotly_chart(fig2, use_container_width=True)

    # --- Export ---
    st.markdown("<div class='section-title' style='margin-top:8px;'>⬇️ Export Results</div>", unsafe_allow_html=True)
    ec1, ec2 = st.columns(2)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    with ec1:
        st.download_button("📥 Download CSV", export_results(df, 'csv'),
                           f"predictions_{ts}.csv", "text/csv", use_container_width=True)
    with ec2:
        st.download_button("📥 Download Excel", export_results(df, 'excel'),
                           f"predictions_{ts}.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           use_container_width=True)


# ==================== HEADER ====================
st.markdown("""
<div class='app-header'>
    <h1>📦 Item Categorization Engine</h1>
    <p>Machine learning–powered hierarchical product classification with confidence scoring</p>
    <div class='status-pill'>● Models Ready</div>
</div>
""", unsafe_allow_html=True)

# ==================== LOAD & TRAIN ====================
loading_placeholder = st.empty()
with loading_placeholder.container():
    prog = st.progress(0, text="Loading training data…")
    training_data, category_pipeline, subcat_models, part_term_models = train_all_models(CSV_URL)
    prog.progress(100, text="✓ All models trained and ready!")
    st.session_state.training_complete = True

loading_placeholder.empty()

# ==================== STATS ROW ====================
with st.expander("📊 Training Data Overview", expanded=False):
    st.markdown(f"""
    <div class='metric-row'>
        <div class='metric-card'>
            <div class='label'>Total Items</div>
            <div class='value'>{len(training_data):,}</div>
        </div>
        <div class='metric-card'>
            <div class='label'>Categories</div>
            <div class='value'>{training_data['Category'].nunique()}</div>
        </div>
        <div class='metric-card'>
            <div class='label'>Subcategories</div>
            <div class='value'>{training_data['Subcategory'].nunique()}</div>
        </div>
        <div class='metric-card'>
            <div class='label'>Part Types</div>
            <div class='value'>{training_data['Part Terminology ID - Name'].nunique()}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    cat_dist = training_data['Category'].value_counts().reset_index()
    cat_dist.columns = ['Category', 'Count']
    fig_overview = px.treemap(cat_dist.head(20), path=['Category'], values='Count',
                              color='Count', color_continuous_scale=['#2e334d', '#6c63ff'])
    fig_overview.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', font_color='#e8eaf6',
        margin=dict(l=0, r=0, t=0, b=0), coloraxis_showscale=False,
    )
    st.plotly_chart(fig_overview, use_container_width=True)

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

# ==================== TABS ====================
tab1, tab2, tab3 = st.tabs(["✍️ Manual Entry", "📤 Batch Upload", "📋 Template & Help"])

# ==================== TAB 1: MANUAL ENTRY ====================
with tab1:
    st.markdown("<div class='section-title'>Enter Item Descriptions</div>", unsafe_allow_html=True)
    st.markdown("<p style='color:var(--text-muted);font-size:14px;margin-bottom:16px;'>One description per line. Get instant category, subcategory, and part terminology predictions with confidence scores.</p>", unsafe_allow_html=True)

    item_input = st.text_area(
        "Item Descriptions",
        height=160,
        placeholder="Stainless steel fastener M8x20\nIndustrial-grade silicone adhesive\nPrecision digital measurement gauge",
        label_visibility='collapsed'
    )

    predict_btn = st.button("🔍 Predict Categories", key='predict_manual', use_container_width=False)

    if predict_btn:
        items = [i.strip() for i in item_input.split('\n') if i.strip()]
        if not items:
            st.markdown("<div class='alert alert-warning'>⚠️ Please enter at least one item description.</div>", unsafe_allow_html=True)
        else:
            if len(items) > MAX_BATCH_SIZE:
                st.markdown(f"<div class='alert alert-warning'>⚠️ Capped at {MAX_BATCH_SIZE} items.</div>", unsafe_allow_html=True)
                items = items[:MAX_BATCH_SIZE]

            bar = st.progress(0, text=f"Processing {len(items)} item(s)…")
            results_list = []
            for idx, item in enumerate(items):
                results_list.append(hierarchical_prediction(item, category_pipeline, subcat_models, part_term_models))
                bar.progress(int((idx + 1) / len(items) * 100), text=f"Processing {idx+1}/{len(items)}…")
            bar.empty()

            df_manual = predictions_to_df(items, results_list)
            st.session_state.last_predictions = df_manual

            st.markdown("<div class='alert alert-success'>✓ Predictions complete!</div>", unsafe_allow_html=True)
            render_results_section(df_manual, mode='manual')

# ==================== TAB 2: BATCH UPLOAD ====================
with tab2:
    st.markdown("<div class='section-title'>Batch Upload</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='alert alert-info'>
    Upload an Excel file with <strong>Item Number</strong> and <strong>Description</strong> columns. Max 5,000 rows.
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Select Excel File (.xlsx / .xls)", type=["xlsx", "xls"])

    if uploaded_file:
        try:
            input_data = pd.read_excel(uploaded_file)
            missing_cols = [c for c in ['Item Number', 'Description'] if c not in input_data.columns]

            if missing_cols:
                st.markdown(f"<div class='alert alert-danger'>❌ Missing columns: {', '.join(missing_cols)}</div>", unsafe_allow_html=True)
            else:
                with st.expander("👀 Preview Uploaded Data", expanded=True):
                    st.dataframe(input_data.head(10), use_container_width=True)
                    st.markdown(f"<p style='color:var(--text-muted);font-size:13px;'>{len(input_data)} total rows</p>", unsafe_allow_html=True)

                if st.button("🚀 Process & Predict", use_container_width=False, key='predict_batch'):
                    if len(input_data) > MAX_BATCH_SIZE:
                        st.markdown(f"<div class='alert alert-warning'>⚠️ Processing first {MAX_BATCH_SIZE} rows.</div>", unsafe_allow_html=True)
                        input_data = input_data.head(MAX_BATCH_SIZE).copy()

                    input_data['Description'] = input_data['Description'].fillna('').astype(str)
                    items = input_data['Description'].tolist()
                    item_numbers = input_data['Item Number'].tolist()

                    bar = st.progress(0, text=f"Processing {len(items)} item(s)…")
                    results_list = []
                    for idx, item in enumerate(items):
                        results_list.append(hierarchical_prediction(item, category_pipeline, subcat_models, part_term_models))
                        bar.progress(int((idx + 1) / len(items) * 100), text=f"Processing {idx+1}/{len(items)}…")
                    bar.empty()

                    df_batch = predictions_to_df(items, results_list, has_item_number=True, item_numbers=item_numbers)
                    st.session_state.last_predictions = df_batch

                    st.markdown("<div class='alert alert-success'>✓ Batch processing complete!</div>", unsafe_allow_html=True)
                    render_results_section(df_batch, mode='batch')

        except Exception as e:
            st.markdown(f"<div class='alert alert-danger'>❌ Error reading file: {str(e)}</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='alert alert-info'>ℹ️ No file selected. Download a template from the <strong>Template & Help</strong> tab.</div>", unsafe_allow_html=True)

# ==================== TAB 3: TEMPLATE & HELP ====================
with tab3:
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("<div class='section-title'>📋 Download Template</div>", unsafe_allow_html=True)
        st.markdown("""
        <ul style='color:var(--text-muted);font-size:14px;line-height:1.8;'>
            <li><strong style='color:var(--text);'>Item Number</strong> — unique identifier</li>
            <li><strong style='color:var(--text);'>Description</strong> — detailed product description</li>
        </ul>
        """, unsafe_allow_html=True)
        st.download_button("📥 Download Excel Template", generate_template(),
                           "categorization_template.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           use_container_width=True)

    with c2:
        st.markdown("<div class='section-title'>🎯 Tips for Best Results</div>", unsafe_allow_html=True)
        st.markdown("""
        <ul style='color:var(--text-muted);font-size:14px;line-height:1.8;'>
            <li>Use <strong style='color:var(--text);'>detailed</strong> product descriptions</li>
            <li>Include <strong style='color:var(--text);'>specs, materials, and measurements</strong></li>
            <li>Avoid very short or vague descriptions</li>
            <li>Consistent terminology improves accuracy</li>
            <li>High confidence (>70%) = reliable prediction</li>
        </ul>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<div class='section-title'>❓ FAQ</div>", unsafe_allow_html=True)

    faqs = [
        ("What predictions do I get?", "A primary category with confidence score, three subcategory options with confidence scores, and three part terminology options with confidence scores."),
        ("How are confidence scores calculated?", "Confidence is derived from the SVM model's decision function scores, normalized via softmax. Higher is more reliable."),
        ("How accurate are predictions?", "Accuracy depends on how closely the description matches training data. Detailed, specific descriptions yield the best results."),
        ("What's the batch limit?", f"Maximum {MAX_BATCH_SIZE} items per batch."),
        ("Why is startup slow?", "Models train on first load and are cached — subsequent runs are instant."),
        ("Which file formats work?", "Both .xlsx and .xls are supported for batch upload."),
    ]
    for q, a in faqs:
        with st.expander(q):
            st.markdown(f"<p style='color:var(--text-muted);font-size:14px;'>{a}</p>", unsafe_allow_html=True)

st.markdown("""
<div style='text-align:center;color:#3a3f5c;font-size:12px;padding:32px 0 16px;font-family:Space Mono,monospace;'>
    ITEM CATEGORIZATION ENGINE · BUILT WITH STREAMLIT & SKLEARN
</div>
""", unsafe_allow_html=True)
