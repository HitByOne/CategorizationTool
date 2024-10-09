import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import io

# Automatically open in wide mode
st.set_page_config(layout="wide")

# Step 1: Connect to Google Sheets and load the ISN Category List
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name('credentials.json', scope)
client = gspread.authorize(creds)

# Load ISN Category List from Google Sheets
sheet_url = 'https://docs.google.com/spreadsheets/d/1u2r5fRh0sEXXkudSvwyS41rRzF-LIQjE/edit?usp=sharing&ouid=101090486103714461716&rtpof=true&sd=true'
sheet = client.open_by_url(sheet_url)
worksheet = sheet.get_worksheet(0)
data = worksheet.get_all_records()
isn_category_data = pd.DataFrame(data)

# Create a mapping from Subcategory to Subcategory-ID
subcategory_id_mapping = isn_category_data.set_index('Subcategory')['Subcategory-ID'].to_dict()

# Step 2: Load the training data from Google Sheets (similar to ISN Category List)
training_sheet_url = 'YOUR_TRAINING_DATA_GOOGLE_SHEET_URL'
training_sheet = client.open_by_url(training_sheet_url)
training_worksheet = training_sheet.get_worksheet(0)
training_data = pd.DataFrame(training_worksheet.get_all_records())

# Handle missing values in the item_name and Subcategory columns
training_data = training_data.dropna(subset=['item_name', 'Subcategory'])

# Step 3: Preprocess and vectorize the item descriptions
X_train = training_data['item_name']  # The item descriptions
y_train = training_data['Subcategory']  # The corresponding categories
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)

# Step 4: Train the Naive Bayes model
model = MultinomialNB()
model.fit(X_train_vectorized, y_train)

# Step 5: Define a function to predict the top three Subcategory-IDs for new items
def predict_top_three_subcategory_ids(item_description):
    X_new = vectorizer.transform([item_description])
    predicted_proba = model.predict_proba(X_new)[0]
    top_three_indices = predicted_proba.argsort()[-3:][::-1]  # Get the top three predictions
    
    # Get the Subcategory-IDs for the top three predicted categories
    top_three_categories = [model.classes_[i] for i in top_three_indices]
    top_three_subcategory_ids = [subcategory_id_mapping.get(cat, 'N/A') for cat in top_three_categories]
    
    return top_three_subcategory_ids

# Step 6: Streamlit App UI
st.title("Item Categorization with Auto-Suggestion")

# Option 1: Textbox input for manual search
st.header("Option 1: Manual Entry")
item_input = st.text_area("Enter item names (one per line)", value="", height=150)
items = [item.strip() for item in item_input.split("\n") if item.strip()]

if st.button("Get Category Suggestions for Manual Entry"):
    if items:
        # Create a DataFrame for entered items
        df = pd.DataFrame({'Item': items})
        
        # Auto-Suggest Top Three Subcategory-IDs based on manual input
        df['Suggested Subcategory-IDs'] = df['Item'].apply(lambda x: predict_top_three_subcategory_ids(x))
        
        # Expand the result to display Subcategory-IDs for each suggestion
        expanded_df = pd.DataFrame(df['Suggested Subcategory-IDs'].to_list(), 
                                   index=df.index, 
                                   columns=['Subcategory-ID 1', 'Subcategory-ID 2', 'Subcategory-ID 3'])
        
        df = pd.concat([df, expanded_df], axis=1)

        # Display the categorized DataFrame
        st.write("### Final Categorized Items")
        st.dataframe(df)

        # Export categorized data
        def convert_df(df):
            return df.to_csv(index=False).encode('utf-8')

        csv = convert_df(df)
        st.download_button(
            "Download Categorized Data",
            csv,
            "categorized_items_manual.csv",
            "text/csv",
            key='download-csv-manual'
        )
    else:
        st.warning("Please enter some item names before pressing the button.")

# Option 2: File upload for batch search
st.header("Option 2: Upload Excel File")

# Download template for Excel upload
st.subheader("Download Excel Template")
def generate_template():
    # Create a template with 'Item Number' and 'Description' columns
    template_df = pd.DataFrame(columns=['Item Number', 'Description'])
    
    # Convert template DataFrame to Excel in-memory
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        template_df.to_excel(writer, index=False)
    buffer.seek(0)
    return buffer

st.download_button(
    label="Download Template",
    data=generate_template(),
    file_name="item_categorization_template.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

# File upload section
uploaded_file = st.file_uploader("Upload an Excel file with 'Item Number' and 'Description' columns", type="xlsx")

if uploaded_file is not None:
    # Load the uploaded file into a DataFrame
    input_data = pd.read_excel(uploaded_file)
    
    # Check if the necessary columns are in the file
    if 'Item Number' in input_data.columns and 'Description' in input_data.columns:
        st.write("### Uploaded Data")
        st.dataframe(input_data.head())
        
        # Auto-Suggest Top Three Subcategory-IDs based on the Description
        input_data['Suggested Subcategory-IDs'] = input_data['Description'].apply(lambda x: predict_top_three_subcategory_ids(x))
        
        # Expand the result to display Subcategory-IDs for each suggestion
        expanded_df = pd.DataFrame(input_data['Suggested Subcategory-IDs'].to_list(), 
                                   index=input_data.index, 
                                   columns=['Subcategory-ID 1', 'Subcategory-ID 2', 'Subcategory-ID 3'])
        
        # Merge the results back into the original data
        final_df = pd.concat([input_data[['Item Number', 'Description']], expanded_df], axis=1)

        # Display the final categorized DataFrame
        st.write("### Final Categorized Items from Excel")
        st.dataframe(final_df)

        # Export categorized data
        def convert_df(df):
            return df.to_csv(index=False).encode('utf-8')

        csv = convert_df(final_df)
        st.download_button(
            "Download Categorized Data",
            csv,
            "categorized_items_excel.csv",
            "text/csv",
            key='download-csv-excel'
        )
    else:
        st.warning("The uploaded file must contain 'Item Number' and 'Description' columns.")
else:
    st.info("Please upload an Excel file to get started.")
