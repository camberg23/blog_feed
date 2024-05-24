import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

API_KEY = st.secrets['API_KEY']

# Initialize OpenAI client with your API key
client = OpenAI(api_key=API_KEY)

def get_embedding(text, model="text-embedding-3-small"):
    response = client.embeddings.create(
        model=model,
        input=text,
        encoding_format="float"
    )
    return response.data[0].embedding
    
def find_top_n_similar_texts(input_text, df, n=5, content_preview_length=100, title_search=None):
    # Filter the DataFrame based on title search if provided
    if title_search:
        df = df[df['title'].str.contains(title_search, case=False, na=False)]
    
    if df.empty:
        return ["No matches found for the provided title search."]
    
    # Generate embedding for the input text
    input_embedding = get_embedding(input_text)
    
    # Function to calculate cosine similarity
    def calculate_cosine_similarity(input_embedding, df_embeddings):
        input_embedding_array = np.array(input_embedding).reshape(1, -1)
        df_embeddings_array = np.stack(df_embeddings)
        similarities = cosine_similarity(input_embedding_array, df_embeddings_array)
        return similarities[0]
    
    # Calculate similarities
    content_similarities = calculate_cosine_similarity(input_embedding, df['content_embedding'])
    
    # Find top N indices for content
    top_n_content_indices = np.argsort(content_similarities)[-n:][::-1]
    
    # Retrieve titles, URLs, and content for the top N indices
    top_n_content_titles = df.iloc[top_n_content_indices]['title'].tolist()
    top_n_content_urls = df.iloc[top_n_content_indices]['url'].tolist()
    top_n_contents = df.iloc[top_n_content_indices]['content'].apply(lambda x: x[:content_preview_length]).tolist()
    
    # Prepare the output list with title, URL, and content
    output_list = []
    for index, (title, url, content) in enumerate(zip(top_n_content_titles, top_n_content_urls, top_n_contents), start=1):
        output_list.append(f"{index}. **[{title}]({url})**\n**Content Preview**: {content}...")
    
    return output_list

# Load the existing DataFrame with embeddings
blog_df = pd.read_pickle('blog_df.pkl')

# Streamlit App
st.title("Blog Feed Tool")

# User inputs
themes = ["Relationships", "Work", "Romance", "Health", "Finance", "Personal Development", "Hobbies", "Technology", "Education", "Travel", "Food", "Lifestyle", "Parenting", "Fitness", "Mental Health"]
selected_themes = st.multiselect("Select Themes", themes)
custom_text = st.text_input("Add Custom Phrases (Optional)", "")
title_search = st.text_input("Title Search (Optional)", "")
n = st.slider("Number of Results", min_value=1, max_value=20, value=5)
content_preview_length = st.slider("Content Preview Length", min_value=50, max_value=500, value=100)

# Combine selected themes and custom text for input text
input_text = " ".join(selected_themes) + " " + custom_text

# Find top n similar texts
if st.button("Generate Blog Feed"):
    top_n_content_list = find_top_n_similar_texts(input_text, blog_df, n, content_preview_length, title_search)
    for item in top_n_content_list:
        st.markdown(item)
        st.markdown("---")
