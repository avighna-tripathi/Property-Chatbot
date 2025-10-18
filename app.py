# app.py
# Author: [Your Name]
# Date: October 18, 2025
# Description: A robust, production-ready chatbot for searching real estate listings.
# This version has been meticulously corrected to match the specific CSV data schema provided.

import streamlit as st
import pandas as pd
import re
import numpy as np
from sentence_transformers import SentenceTransformer, util
import time

# --- Configuration & Constants ---
# File paths are centralized for easy management.
PROJECT_DATA_PATH = 'data/project.csv'
ADDRESS_DATA_PATH = 'data/ProjectAddress.csv'
CONFIG_DATA_PATH = 'data/ProjectConfiguration.csv'
VARIANT_DATA_PATH = 'data/ProjectConfigurationVariant.csv'
SBERT_MODEL_NAME = 'all-MiniLM-L6-v2'


# --- Data Loading & Preparation ---
# This function is the heart of the data pipeline. It's designed to be executed
# only once and its result is cached for performance.
@st.cache_data
def get_property_data():
    """
    Reads all four property-related CSVs, performs the correct joins based on their keys,
    and engineers features for searching. This function is critical and has been
    written to be robust against missing data.
    """
    try:
        projects = pd.read_csv(PROJECT_DATA_PATH)
        addresses = pd.read_csv(ADDRESS_DATA_PATH)
        configs = pd.read_csv(CONFIG_DATA_PATH)
        variants = pd.read_csv(VARIANT_DATA_PATH)
    except FileNotFoundError as e:
        st.error(f"Fatal Error: A data file is missing. Please check the `data` directory. Details: {e}")
        return None

    # --- Data Merging Strategy ---
    # The merge strategy is crucial. We start with the most granular data (variants)
    # and progressively join larger tables to it. Keys are explicitly renamed to
    # a consistent format ('PROJECT_ID', 'CONFIG_ID') to prevent ambiguity.

    # 1. Standardize primary/foreign keys for reliable merging.
    projects.rename(columns={'id': 'PROJECT_ID'}, inplace=True)
    addresses.rename(columns={'projectId': 'PROJECT_ID'}, inplace=True)
    configs.rename(columns={'id': 'CONFIG_ID', 'projectId': 'PROJECT_ID'}, inplace=True)
    variants.rename(columns={'configurationId': 'CONFIG_ID'}, inplace=True)

    # 2. Merge variants with configurations to link price/area to a BHK type.
    # An 'inner' join ensures we only have records that have both config and variant info.
    merged_configs = pd.merge(variants, configs, on='CONFIG_ID', how='inner')

    # 3. Merge the result with the main project data (projectName, status, etc.).
    merged_projects = pd.merge(merged_configs, projects, on='PROJECT_ID', how='left')

    # 4. Finally, join the address data to get the full location details.
    # A 'left' join ensures we don't lose projects if an address is missing.
    prop_df = pd.merge(merged_projects, addresses, on='PROJECT_ID', how='left')

    # --- Feature Engineering & Cleaning ---
    # Create a unified text field for semantic search. Fill NaNs to prevent errors.
    prop_df['SEARCH_TEXT'] = (
        prop_df['projectName'].fillna('') + ' apartment located at ' +
        prop_df['fullAddress'].fillna('') + '. This is a ' +
        prop_df['type'].fillna('') + ' unit. The project is currently ' +
        prop_df['status'].fillna('') + '.'
    )

    # Convert key columns to numeric types for filtering, coercing errors to NaN.
    prop_df['PRICE_NUM'] = pd.to_numeric(prop_df['price'], errors='coerce').fillna(0).astype(np.int64)
    prop_df['BHK_NUM'] = prop_df['type'].str.extract(r'(\d+)').astype(float).fillna(0).astype(int)

    # Ensure carpetArea is also numeric and clean for display.
    prop_df['carpetArea'] = pd.to_numeric(prop_df['carpetArea'], errors='coerce').fillna(0)
    
    return prop_df

@st.cache_resource
def load_sbert_model():
    """Loads and caches the Sentence Transformer model."""
    with st.spinner("Warming up the AI... This might take a moment."):
        model = SentenceTransformer(SBERT_MODEL_NAME)
    return model

@st.cache_data
def create_embeddings(_df, _model):
    """Generates sentence embeddings for the SEARCH_TEXT column."""
    st.info("Creating semantic embeddings for the property data...", icon="🧠")
    embeddings = _model.encode(_df['SEARCH_TEXT'].tolist(), convert_to_tensor=True, show_progress_bar=True)
    st.success("Embeddings created successfully!", icon="✅")
    time.sleep(1.5)
    return embeddings


# --- Helper Functions & Core Logic ---

def format_price(price):
    """Helper to format a numeric price into a readable string (Crores or Lakhs)."""
    if not pd.isna(price) and price >= 10000000:
        return f"₹ {price / 10000000:.2f} Cr"
    elif not pd.isna(price):
        return f"₹ {price / 100000:.2f} L"
    return "Price not available"

def extract_filters_from_query(query_text):
    """
    Uses regex to pull out structured information like BHK, price, and location.
    """
    filters = {}
    bhk_pattern = re.search(r'(\d+)\s*bhk', query_text, re.IGNORECASE)
    if bhk_pattern:
        filters['bhk'] = int(bhk_pattern.group(1))

    budget_pattern = re.search(r'(under|below|in|max)\s*([\d.]+)\s*(cr|lakh|l)', query_text, re.IGNORECASE)
    if budget_pattern:
        amount = float(budget_pattern.group(2))
        unit = budget_pattern.group(3).lower()
        if unit == 'cr':
            filters['max_price'] = amount * 10000000
        else:
            filters['max_price'] = amount * 100000
    
    loc_pattern = re.search(r'in\s+([\w\s,]+)', query_text, re.IGNORECASE)
    if loc_pattern:
        loc_string = loc_pattern.group(1).strip().lower()
        filters['location_terms'] = [term.strip() for term in loc_string.replace(',', ' ').split()]
        
    return filters

def find_properties(df, query, model, embeddings):
    """
    The main search function. It combines rule-based filtering with semantic search.
    """
    filters = extract_filters_from_query(query)
    filtered_df = df.copy()

    if 'bhk' in filters:
        filtered_df = filtered_df[filtered_df['BHK_NUM'] == filters['bhk']]
    if 'max_price' in filters:
        filtered_df = filtered_df[filtered_df['PRICE_NUM'] <= filters['max_price']]
    if 'location_terms' in filters:
        for term in filters['location_terms']:
            filtered_df = filtered_df[
                filtered_df['projectName'].str.lower().str.contains(term, na=False) |
                filtered_df['fullAddress'].str.lower().str.contains(term, na=False)
            ]

    if not filtered_df.empty:
        result_indices = filtered_df.index.tolist()
        corpus_embeddings = embeddings[result_indices]
        query_embedding = model.encode(query, convert_to_tensor=True)
        cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
        top_indices = np.argsort(-cos_scores.cpu().numpy())
        sorted_indices = [result_indices[i] for i in top_indices]
        ranked_df = df.loc[sorted_indices]
        return ranked_df.head(5)
    
    return pd.DataFrame()


# --- UI Components ---
def display_property_card(prop):
    """
    Renders a single property result in a card format.
    Uses .get() for all fields to prevent KeyErrors if data is unexpectedly missing.
    """
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(prop.get('projectName', 'N/A'))
        st.caption(prop.get('fullAddress', 'Address not available'))
        st.markdown(f"**Type:** {prop.get('type', 'N/A')}")
        st.markdown(f"**Status:** `{prop.get('status', 'N/A')}`")

    with col2:
        st.subheader(format_price(prop.get('PRICE_NUM')))
        st.markdown(f"**Carpet Area:** {prop.get('carpetArea', 0):.0f} sq.ft.")

    st.link_button("More Details", f"https://www.nobroker.in/project/{prop.get('slug', '')}")


# --- Main Application ---
def main():
    """The main function that runs the Streamlit app."""
    st.set_page_config(page_title="Real Estate AI", page_icon="🏘️")
    st.title("🏘️ AI-Powered Property Finder")
    st.markdown("Ask me to find properties, like _'show me 3bhk flats under 2cr in Pune'_.")

    property_df = get_property_data()
    
    if property_df is None:
        st.warning("Data could not be loaded. The app cannot proceed.")
        st.stop()
        
    sbert_model = load_sbert_model()
    corpus_embeddings = create_embeddings(property_df, sbert_model)

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help you find a new home today?"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "cards" in message and not message["cards"].empty:
                for index, prop in message["cards"].iterrows():
                    display_property_card(prop)

    if user_query := st.chat_input("Your query..."):
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.spinner("Searching thousands of listings for you..."):
            results = find_properties(property_df, user_query, sbert_model, corpus_embeddings)

        if not results.empty:
            min_price = results['PRICE_NUM'].min()
            top_project_name = results.iloc[0].get('projectName', 'various projects')
            
            summary = (f"Okay, I found {len(results)} great options for you! "
                       f"Top results include listings in projects like **{top_project_name}**, "
                       f"with prices starting from around **{format_price(min_price)}**. "
                       "Here are the best matches:")
            
            assistant_response = {"role": "assistant", "content": summary, "cards": results}
        else:
            summary = ("Apologies, I couldn't find any properties matching those exact criteria. "
                       "Could you try being more general? For example, try a different location or a wider budget.")
            assistant_response = {"role": "assistant", "content": summary, "cards": pd.DataFrame()}
            
        with st.chat_message("assistant"):
            st.markdown(assistant_response["content"])
            if not assistant_response["cards"].empty:
                for index, prop in assistant_response["cards"].iterrows():
                    display_property_card(prop)
        
        st.session_state.messages.append(assistant_response)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        st.error("An unexpected error occurred. Please check the console for details.")
        st.exception(e)

