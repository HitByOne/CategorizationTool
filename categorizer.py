import streamlit as st
import json
from anthropic import Anthropic

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Smart Item Categorizer",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🎯 Smart Item Categorizer")
st.markdown("Define your categories once, let AI categorize items intelligently")

# ==================== INITIALIZE ANTHROPIC ====================
client = Anthropic()

# ==================== SESSION STATE ====================
if 'categories' not in st.session_state:
    st.session_state.categories = {}
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

# ==================== SIDEBAR: DEFINE CATEGORIES ====================
st.sidebar.header("📋 Define Your Categories")

with st.sidebar:
    st.markdown("### Step 1: Create Your Category Structure")
    
    num_categories = st.number_input(
        "How many main categories?",
        min_value=1,
        max_value=20,
        value=3,
        key="num_cat"
    )
    
    categories = {}
    for i in range(num_categories):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            cat_name = st.text_input(
                f"Category {i+1} name",
                value=f"Category {i+1}",
                key=f"cat_name_{i}"
            )
        
        with col2:
            num_subcats = st.number_input(
                "Subcats",
                min_value=0,
                max_value=10,
                value=2,
                key=f"num_subcat_{i}"
            )
        
        if cat_name:
            subcategories = []
            for j in range(num_subcats):
                subcat = st.text_input(
                    f"  └─ Subcategory {j+1}",
                    value=f"Sub {j+1}",
                    key=f"subcat_{i}_{j}",
                    label_visibility="collapsed"
                )
                if subcat:
                    subcategories.append(subcat)
            
            categories[cat_name] = subcategories
    
    st.session_state.categories = categories
    
    # Add descriptions for each category
    st.markdown("### Step 2: Add Category Descriptions")
    st.info("Optional: Add descriptions to help AI understand each category better")
    
    category_descriptions = {}
    for cat_name in categories.keys():
        desc = st.text_area(
            f"What is '{cat_name}'?",
            placeholder="e.g., 'Items used to connect or attach things together'",
            height=60,
            key=f"desc_{cat_name}"
        )
        if desc:
            category_descriptions[cat_name] = desc
    
    st.session_state.category_descriptions = category_descriptions

# ==================== MAIN CONTENT ====================
if not st.session_state.categories:
    st.warning("⚠️ Please define at least one category in the sidebar first!")
else:
    # Show category structure
    with st.expander("📊 Your Category Structure", expanded=True):
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
        category_descriptions = getattr(st.session_state, 'category_descriptions', {})
        
        system_prompt = f"""You are an expert categorization assistant. Your job is to categorize items based on the following structure:

CATEGORY STRUCTURE:
{category_structure}

CATEGORY DESCRIPTIONS:
{json.dumps(category_descriptions, indent=2) if category_descriptions else "No additional descriptions provided"}

IMPORTANT RULES:
1. You MUST respond with valid JSON only, no other text
2. Each item MUST be categorized into EXACTLY ONE main category and ONE subcategory
3. If you're unsure, choose the BEST match based on the item's primary purpose
4. Response format MUST be:
{{"item": "item name", "category": "main category", "subcategory": "subcategory", "confidence": 0.0-1.0, "reason": "brief explanation"}}

Examples of valid responses:
{{"item": "Steel Bolt", "category": "Fasteners", "subcategory": "Bolts", "confidence": 0.95, "reason": "Metal fastening device"}}
{{"item": "Wood Glue", "category": "Adhesives", "subcategory": "Epoxy", "confidence": 0.8, "reason": "Adhesive for bonding"}}

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
                    
                    # Display result with nice formatting
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Item", result.get("item", "N/A"))
                    
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
                        if st.button("✅ Yes, this is correct", use_container_width=True):
                            st.success("Great! AI learns from correct categorizations")
                    
                    with col2:
                        if st.button("❌ No, let me correct", use_container_width=True):
                            st.session_state.show_correction = True
                    
                    with col3:
                        if st.button("🔄 Try again", use_container_width=True):
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
                            correction_msg = f"You were wrong. '{item_input}' should be categorized as '{correct_category}' / '{correct_subcategory}'"
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
                
                except json.JSONDecodeError:
                    st.error("❌ Failed to parse response. Please try again.")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        # Batch upload
        st.markdown("---")
        st.subheader("📤 Batch Categorize")
        
        uploaded_file = st.file_uploader("Upload CSV file (with 'Item' column)", type=["csv"])
        
        if uploaded_file:
            try:
                import pandas as pd
                df = pd.read_csv(uploaded_file)
                
                if 'Item' not in df.columns:
                    st.error("❌ CSV must have an 'Item' column")
                else:
                    items_list = df['Item'].tolist()
                    
                    if st.button("🚀 Categorize All", use_container_width=True):
                        results = []
                        progress_bar = st.progress(0)
                        
                        for idx, item in enumerate(items_list):
                            try:
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
                                    "item": item,
                                    "category": "Error",
                                    "subcategory": str(e),
                                    "confidence": 0,
                                    "reason": "Failed to categorize"
                                })
                            
                            progress_bar.progress((idx + 1) / len(items_list))
                        
                        # Display results
                        st.success(f"✓ Categorized {len(results)} items")
                        
                        results_df = pd.DataFrame(results)
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Download results
                        csv = results_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "📥 Download Results",
                            csv,
                            "categorized_items.csv",
                            "text/csv",
                            use_container_width=True
                        )
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
    
    with tab2:
        st.subheader("💬 Conversation History")
        
        if not st.session_state.conversation_history:
            st.info("No conversation yet. Start categorizing items above!")
        else:
            for message in st.session_state.conversation_history:
                if message["role"] == "user":
                    st.markdown(f"**You:** {message['content']}")
                else:
                    st.markdown(f"**AI:** {message['content']}")
            
            if st.button("🗑️ Clear History"):
                st.session_state.conversation_history = []
                st.rerun()

# ==================== FOOTER ====================
st.divider()
st.markdown("""
<div style='text-align: center; color: #999; font-size: 12px; padding: 20px;'>
Smart Item Categorizer | Powered by Claude AI | 
<a href='https://anthropic.com' target='_blank'>Anthropic</a>
</div>
""", unsafe_allow_html=True)
