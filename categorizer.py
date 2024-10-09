import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import gdown
import io

# Automatically open in wide mode
st.set_page_config(layout="wide")

# Step 1: Download the training data from Google Drive
file_url = 'https://drive.google.com/uc?id=1MmnakF0kEnN5t-E3_RCuKlZjmsCfhFtv'
output_file = '/tmp/training_data.csv'  # Local path to save the file

# Download the file using gdown
gdown.download(file_url, output_file, quiet=False)

# Step 2: Load the training data
training_data = pd.read_csv(output_file)

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
    
    return top_three_categories

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
