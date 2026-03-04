import streamlit as st
import json
import pandas as pd
from anthropic import Anthropic
from io import BytesIO

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Smart Item Categorizer",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🎯 Smart Item Categorizer")
st.markdown("Use your existing category structure with AI-powered categorization")

# ==================== INITIALIZE ANTHROPIC ====================
client = Anthropic()

# ==================== SESSION STATE ====================
if 'categories' not in st.session_state:
    st.session_state.categories = {}
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'category_descriptions' not in st.session_state:
    st.session_state.category_descriptions = {}

# ==================== SIDEBAR: LOAD CATEGORIES ====================
st.sidebar.header("📋 Load Your Categories")

with st.sidebar:
    st.markdown("### Step 1: Upload Your Excel File")
    st.info("""
    **Expected format:**
    - **Column 1:** Category names
    - **Column 2+:** Subcategory names
    
    Example:
    | Category | Sub1 | Sub2 | Sub3 |
    |----------|------|------|------|
    | Fasteners | Bolts | Nuts | Screws |
    | Adhesives | Epoxy | Glue | Tape |
    """)
    
    uploaded_file = st.file_uploader(
        "Upload Excel file",
        type=["xlsx", "xls"],
        key="category_file"
    )
    
    if uploaded_file:
        try:
            # Read the Excel file
            df = pd.read_excel(uploaded_file)
            
            # Parse the structure
            categories = {}
            for idx, row in df.iterrows():
                cat_name = row.iloc[0]  # First column is category name
                
                if pd.isna(cat_name):
                    continue
                
                cat_name = str(cat_name).strip()
                
                # Get all non-empty subcategories
                subcats = []
                for col_idx in range(1, len(row)):
                    subcat = row.iloc[col_idx]
                    if pd.notna(subcat):
                        subcat_str = str(subcat).strip()
                        if subcat_str and subcat_str.lower() != 'nan':
                            subcats.append(subcat_str)
                
                if cat_name:
                    categories[cat_name] = subcats
            
            st.session_state.categories = categories
            
            st.success(f"✅ Loaded {len(categories)} categories")
            
            # Show what was loaded
            with st.expander("📊 Loaded Structure", expanded=True):
                for cat_name, subcats in categories.items():
                    st.markdown(f"**{cat_name}**")
                    for subcat in subcats:
                        st.markdown(f"  • {subcat}")
        
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
            st.info("Make sure the Excel file has categories in the first column and subcategories in subsequent columns")
    
    st.markdown("---")
    st.markdown("### Step 2: (Optional) Add Category Descriptions")
    st.info("Descriptions help AI understand each category better")
    
    category_descriptions = {}
    if st.session_state.categories:
        for cat_name in st.session_state.categories.keys():
            desc = st.text_area(
                f"What is '{cat_name}'?",
                placeholder="e.g., 'Items used to connect or attach things together'",
                height=50,
                key=f"desc_{cat_name}"
            )
            if desc:
                category_descriptions[cat_name] = desc
        
        st.session_state.category_descriptions = category_descriptions

# ==================== MAIN CONTENT ====================
if not st.session_state.categories:
    st.warning("⚠️ Please upload your Excel file in the sidebar first!")
    
    # Show example
    st.markdown("### 📋 Example Excel Format")
    
    example_data = {
        'Category': ['Fasteners', 'Adhesives', 'Tools', 'Seals'],
        'Sub1': ['Bolts', 'Epoxy', 'Measuring', 'Gaskets'],
        'Sub2': ['Nuts', 'Cyanoacrylate', 'Cutting', 'O-Rings'],
        'Sub3': ['Screws', 'Hot Melt', 'Assembly', 'Seals'],
        'Sub4': ['Anchors', '', '', '']
    }
    
    example_df = pd.DataFrame(example_data)
    st.dataframe(example_df, use_container_width=True)
    
    st.markdown("""
    **How to prepare your Excel file:**
    1. Put category names in the first column
    2. Put subcategories in columns 2, 3, 4, etc.
    3. Leave empty cells if a category has fewer subcategories
    4. Save as .xlsx or .xls
    5. Upload above
    """)

else:
    # Show category structure
    with st.expander("📊 Your Category Structure", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Categories")
            for cat_name, subcats in st.session_state.categories.items():
                st.markdown(f"**{cat_name}**")
                for subcat in subcats:
                    st.markdown(f"  • {subcat}")
        
        with col2:
            st.markdown("### JSON Format")
            st.code(json.dumps(st.session_state.categories, indent=2), language="json")
    
    st.divider()
    
    # ==================== CATEGORIZATION INTERFACE ====================
    tab1, tab2 = st.tabs(["🔍 Categorize Items", "💬 Chat History"])
    
    with tab1:
        st.subheader("Enter Items to Categorize")
        
        # Build the system prompt
        category_structure = json.dumps(st.session_state.categories, indent=2)
        category_descriptions = st.session_state.category_descriptions
        
        system_prompt = f"""You are an expert categorization assistant. Your job is to categorize items based on the following structure:

CATEGORY STRUCTURE:
{category_structure}

CATEGORY DESCRIPTIONS:
{json.dumps(category_descriptions, indent=2) if category_descriptions else "No additional descriptions provided"}

IMPORTANT RULES:
1. You MUST respond with valid JSON only, no other text
2. Each item MUST be categorized into EXACTLY ONE main category and ONE subcategory
3. If you're unsure, choose the BEST match based on the item's primary purpose
4. The category and subcategory MUST exist in the structure above
5. Response format MUST be:
{{"item": "item name", "category": "main category", "subcategory": "subcategory", "confidence": 0.0-1.0, "reason": "brief explanation"}}

Examples of valid responses:
{{"item": "Steel Bolt", "category": "Fasteners", "subcategory": "Bolts", "confidence": 0.95, "reason": "Metal fastening device"}}
{{"item": "Wood Glue", "category": "Adhesives", "subcategory": "Hot Melt", "confidence": 0.8, "reason": "Adhesive for bonding wood"}}

NEVER respond with anything other than valid JSON."""

        col1, col2 = st.columns([4, 1])
        
        with col1:
            item_input = st.text_input(
                "Enter an item name or description:",
                placeholder="e.g., Stainless steel hex bolt 10mm",
                label_visibility="collapsed"
            )
        
        with col2:
            categorize_btn = st.button("📌 Categorize", use_container_width=True)
        
        if categorize_btn and item_input:
            with st.spinner("🤔 Analyzing..."):
                try:
                    # Add user message to history
                    st.session_state.conversation_history.append({
                        "role": "user",
                        "content": f"Categorize this item: {item_input}"
                    })
                    
                    # Get response from Claude
                    response = client.messages.create(
                        model="claude-3-5-sonnet-20241022",
                        max_tokens=500,
                        system=system_prompt,
                        messages=st.session_state.conversation_history
                    )
                    
                    assistant_message = response.content[0].text
                    
                    # Add assistant response to history
                    st.session_state.conversation_history.append({
                        "role": "assistant",
                        "content": assistant_message
                    })
                    
                    # Parse the JSON response
                    result = json.loads(assistant_message)
                    
                    # Validate the result
                    category = result.get("category", "")
                    subcategory = result.get("subcategory", "")
                    
                    # Check if category exists
                    if category not in st.session_state.categories:
                        st.warning(f"⚠️ AI suggested '{category}' but this category doesn't exist in your structure")
                    elif subcategory not in st.session_state.categories.get(category, []):
                        st.warning(f"⚠️ AI suggested '{subcategory}' but this subcategory doesn't exist under '{category}'")
                    
                    # Display result with nice formatting
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Item", result.get("item", "N/A")[:20])
                    
                    with col2:
                        st.metric("Category", result.get("category", "N/A"))
                    
                    with col3:
                        st.metric("Subcategory", result.get("subcategory", "N/A"))
                    
                    with col4:
                        confidence = result.get("confidence", 0)
                        st.metric("Confidence", f"{int(confidence*100)}%")
                    
                    st.info(f"**Reason:** {result.get('reason', 'N/A')}")
                    
                    # Allow user to correct if wrong
                    st.markdown("### Is this correct?")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if st.button("✅ Yes, correct!", use_container_width=True):
                            st.success("Great! Categorization confirmed")
                            st.session_state.show_correction = False
                    
                    with col2:
                        if st.button("❌ Wrong, let me fix", use_container_width=True):
                            st.session_state.show_correction = True
                    
                    with col3:
                        if st.button("🔄 Clear & try again", use_container_width=True):
                            st.session_state.conversation_history = []
                            st.rerun()
                    
                    # Correction interface
                    if 'show_correction' in st.session_state and st.session_state.show_correction:
                        st.markdown("### Correct the categorization")
                        
                        correct_category = st.selectbox(
                            "Select correct category:",
                            list(st.session_state.categories.keys()),
                            key="correct_cat"
                        )
                        
                        correct_subcategory = st.selectbox(
                            "Select correct subcategory:",
                            st.session_state.categories.get(correct_category, []),
                            key="correct_subcat"
                        )
                        
                        if st.button("✓ Save Correction", use_container_width=True):
                            correction_msg = f"You were wrong. '{item_input}' should be categorized as '{correct_category}' / '{correct_subcategory}'. Learn from this."
                            st.session_state.conversation_history.append({
                                "role": "user",
                                "content": correction_msg
                            })
                            
                            ack_response = client.messages.create(
                                model="claude-3-5-sonnet-20241022",
                                max_tokens=100,
                                system="Acknowledge that you've learned from this correction. Respond with: 'Understood. I'll remember that [item] belongs in [category]/[subcategory].'",
                                messages=st.session_state.conversation_history
                            )
                            
                            st.session_state.conversation_history.append({
                                "role": "assistant",
                                "content": ack_response.content[0].text
                            })
                            
                            st.success("✓ Correction saved! AI is learning")
                            st.session_state.show_correction = False
                            st.rerun()
                
                except json.JSONDecodeError as e:
                    st.error(f"❌ Failed to parse AI response: {str(e)}")
                    st.info("This sometimes happens. Try again or rephrase your item description")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        # Batch upload
        st.markdown("---")
        st.subheader("📤 Batch Categorize Items")
        
        uploaded_file = st.file_uploader(
            "Upload CSV file with items to categorize",
            type=["csv"],
            key="items_file"
        )
        
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                
                # Find the item column (flexible - could be named Item, Product, Name, etc.)
                item_col = None
                for col_name in ['Item', 'Product', 'Name', 'Description', 'Title']:
                    if col_name in df.columns:
                        item_col = col_name
                        break
                
                if not item_col and len(df.columns) > 0:
                    item_col = df.columns[0]  # Use first column if no standard name
                
                if not item_col:
                    st.error("❌ CSV file is empty")
                else:
                    items_list = df[item_col].tolist()
                    st.write(f"**Found {len(items_list)} items to categorize**")
                    
                    # Show preview
                    with st.expander("👀 Preview", expanded=False):
                        st.dataframe(df.head(10), use_container_width=True)
                    
                    if st.button("🚀 Categorize All Items", use_container_width=True):
                        results = []
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        for idx, item in enumerate(items_list):
                            try:
                                status_text.text(f"Processing {idx+1}/{len(items_list)}: {str(item)[:50]}")
                                
                                response = client.messages.create(
                                    model="claude-3-5-sonnet-20241022",
                                    max_tokens=500,
                                    system=system_prompt,
                                    messages=[{
                                        "role": "user",
                                        "content": f"Categorize this item: {item}"
                                    }]
                                )
                                
                                result = json.loads(response.content[0].text)
                                results.append(result)
                            except Exception as e:
                                results.append({
                                    "item": str(item),
                                    "category": "Error",
                                    "subcategory": "Failed",
                                    "confidence": 0,
                                    "reason": str(e)[:100]
                                })
                            
                            progress_bar.progress((idx + 1) / len(items_list))
                        
                        status_text.empty()
                        
                        # Display results
                        st.success(f"✅ Categorized {len(results)} items")
                        
                        results_df = pd.DataFrame(results)
                        st.dataframe(results_df, use_container_width=True, height=400)
                        
                        # Download results
                        csv = results_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "📥 Download Results as CSV",
                            csv,
                            "categorized_items.csv",
                            "text/csv",
                            use_container_width=True
                        )
                        
                        # Summary stats
                        st.markdown("### Summary")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            successful = len([r for r in results if r['category'] != 'Error'])
                            st.metric("Successfully Categorized", successful)
                        
                        with col2:
                            avg_confidence = sum([r.get('confidence', 0) for r in results]) / len(results)
                            st.metric("Avg Confidence", f"{int(avg_confidence*100)}%")
                        
                        with col3:
                            errors = len([r for r in results if r['category'] == 'Error'])
                            st.metric("Errors", errors)
            
            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")
    
    with tab2:
        st.subheader("💬 Conversation History")
        
        if not st.session_state.conversation_history:
            st.info("No conversation yet. Start categorizing items above!")
        else:
            for message in st.session_state.conversation_history:
                if message["role"] == "user":
                    st.markdown(f"**You:** {message['content'][:100]}...")
                else:
                    st.markdown(f"**AI:** {message['content'][:100]}...")
            
            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state.conversation_history = []
                st.rerun()

# ==================== FOOTER ====================
st.divider()
st.markdown("""
<div style='text-align: center; color: #999; font-size: 12px; padding: 20px;'>
Smart Item Categorizer | Upload your Excel structure | Powered by Claude AI
</div>
""", unsafe_allow_html=True)
