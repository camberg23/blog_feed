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
    
def find_top_n_similar_texts(input_text, df, n=5, content_preview_length=100, title_or_personality_search=None):
    # Filter the DataFrame based on title or personality type search if provided
    if title_or_personality_search:
        df = df[df['title'].str.contains(title_or_personality_search, case=False, na=False)]
    
    if df.empty:
        return ["No matches found for the provided search criteria."]
    
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
        output_list.append(f"{index}. **[{title}]({url})**\n\n**Content Preview**: {content}...\n")
    
    return output_list

# Load the existing DataFrame with embeddings
blog_df = pd.read_pickle('blog_df.pkl')

# Streamlit App
st.title("Blog Feed Tool")

# Add explanatory text in an expander
with st.expander("Tool Logic/Directions"):
    st.write("""
    ## Tool Logic
    - **Search Term**: If provided, the tool will prioritize this and ignore themes.
    - **Personality Type Search**: If provided, the tool will filter results to include this exact input in titles.
    - **Number of Results**: Specifies how many results to retrieve per theme if no Search Term is provided.
    - **Toggle**: If enabled, performs `n` searches with and without the personality type search when a Search Term is provided.
    - **Themes**: If no Search Term is provided, the tool will iteratively search for the selected themes. 
        - If a personality type is provided, it will be combined with each theme to filter the results.
        - If no personality type is provided, it will simply search for the themes.
    """)

# Initialize session state variables
if 'themes' not in st.session_state:
    st.session_state.themes = []
if 'custom_text' not in st.session_state:
    st.session_state.custom_text = ""
if 'title_or_personality_search' not in st.session_state:
    st.session_state.title_or_personality_search = ""
if 'n' not in st.session_state:
    st.session_state.n = 5
if 'content_preview_length' not in st.session_state:
    st.session_state.content_preview_length = 350
if 'toggle_title_search' not in st.session_state:
    st.session_state.toggle_title_search = False

# User inputs
themes = ["Relationships", "Work", "Romance", "Health", "Finance", "Personal Development", "Hobbies", "Technology", "Education", "Travel", "Food", "Lifestyle", "Parenting", "Fitness", "Mental Health"]
selected_themes = st.multiselect("Select Themes", themes, default=st.session_state.themes)
custom_text = st.text_input("Search Term (will prioritize this over other inputs)", st.session_state.custom_text)
title_or_personality_search = st.text_input("Personality Type Search (will require blog titles to include this exact input)", st.session_state.title_or_personality_search)

col1, col2 = st.columns([3, 1])
with col1:
    n = st.slider("Number of Results per Theme (if no Search Term)", min_value=1, max_value=20, value=st.session_state.n)
with col2:
    toggle_title_search = st.checkbox("Enable Personality Type Search Toggle", value=st.session_state.toggle_title_search)

content_preview_length = st.slider("Content Preview Length", min_value=50, max_value=500, value=st.session_state.content_preview_length)

# Update session state
if selected_themes:
    st.session_state.themes = selected_themes
if custom_text:
    st.session_state.custom_text = custom_text
if title_or_personality_search:
    st.session_state.title_or_personality_search = title_or_personality_search
st.session_state.n = n
st.session_state.content_preview_length = content_preview_length
st.session_state.toggle_title_search = toggle_title_search

# Combine selected themes and custom text for input text
if custom_text:
    input_text = custom_text
else:
    input_text = " ".join(selected_themes)

# Function to process and filter based on themes and personality type
def process_and_filter(df, themes, personality_type, n, content_preview_length):
    results = []
    for theme in themes:
        input_text = theme
        filtered_results = find_top_n_similar_texts(input_text, df, n, content_preview_length, personality_type)
        results.extend(filtered_results)
    return results

# Generate Blog Feed based on user input
if st.button("Generate Blog Feed"):
    if custom_text:
        if toggle_title_search and title_or_personality_search:
            results_without_title_search = find_top_n_similar_texts(custom_text, blog_df, n, content_preview_length)
            results_with_title_search = find_top_n_similar_texts(custom_text, blog_df, n, content_preview_length, title_or_personality_search)
            top_n_content_list = results_without_title_search + results_with_title_search
        else:
            top_n_content_list = find_top_n_similar_texts(custom_text, blog_df, n, content_preview_length)
    else:
        top_n_content_list = process_and_filter(blog_df, selected_themes, title_or_personality_search, n, content_preview_length)
    
    for item in top_n_content_list:
        st.markdown(item)
        st.markdown("---")

# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
# from openai import OpenAI

# API_KEY = st.secrets['API_KEY']

# # Initialize OpenAI client with your API key
# client = OpenAI(api_key=API_KEY)

# def get_embedding(text, model="text-embedding-3-small"):
#     response = client.embeddings.create(
#         model=model,
#         input=text,
#         encoding_format="float"
#     )
#     return response.data[0].embedding
    
# def find_top_n_similar_texts(input_text, df, n=5, content_preview_length=100, title_search=None):
#     # Filter the DataFrame based on title search if provided
#     if title_search:
#         df = df[df['title'].str.contains(title_search, case=False, na=False)]
    
#     if df.empty:
#         return ["No matches found for the provided title search."]
    
#     # Generate embedding for the input text
#     input_embedding = get_embedding(input_text)
    
#     # Function to calculate cosine similarity
#     def calculate_cosine_similarity(input_embedding, df_embeddings):
#         input_embedding_array = np.array(input_embedding).reshape(1, -1)
#         df_embeddings_array = np.stack(df_embeddings)
#         similarities = cosine_similarity(input_embedding_array, df_embeddings_array)
#         return similarities[0]
    
#     # Calculate similarities
#     content_similarities = calculate_cosine_similarity(input_embedding, df['content_embedding'])
    
#     # Find top N indices for content
#     top_n_content_indices = np.argsort(content_similarities)[-n:][::-1]
    
#     # Retrieve titles, URLs, and content for the top N indices
#     top_n_content_titles = df.iloc[top_n_content_indices]['title'].tolist()
#     top_n_content_urls = df.iloc[top_n_content_indices]['url'].tolist()
#     top_n_contents = df.iloc[top_n_content_indices]['content'].apply(lambda x: x[:content_preview_length]).tolist()
    
#     # Prepare the output list with title, URL, and content
#     output_list = []
#     for index, (title, url, content) in enumerate(zip(top_n_content_titles, top_n_content_urls, top_n_contents), start=1):
#         output_list.append(f"{index}. **[{title}]({url})**\n\n**Content Preview**: {content}...\n")
    
#     return output_list

# # Load the existing DataFrame with embeddings
# blog_df = pd.read_pickle('blog_df.pkl')

# # Streamlit App
# st.title("Blog Feed Tool")

# # Initialize session state variables
# if 'themes' not in st.session_state:
#     st.session_state.themes = []
# if 'custom_text' not in st.session_state:
#     st.session_state.custom_text = ""
# if 'title_search' not in st.session_state:
#     st.session_state.title_search = ""
# if 'n' not in st.session_state:
#     st.session_state.n = 5
# if 'content_preview_length' not in st.session_state:
#     st.session_state.content_preview_length = 350

# # User inputs
# themes = ["Relationships", "Work", "Romance", "Health", "Finance", "Personal Development", "Hobbies", "Technology", "Education", "Travel", "Food", "Lifestyle", "Parenting", "Fitness", "Mental Health"]
# selected_themes = st.multiselect("Select Themes", themes, default=st.session_state.themes)
# custom_text = st.text_input("Any Custom Phrases (will try to search for semantically related content)", st.session_state.custom_text)
# title_search = st.text_input("Any Required Keywords (will require blog titles to include this exact input)", st.session_state.title_search)
# n = st.slider("Number of Results", min_value=1, max_value=20, value=st.session_state.n)
# content_preview_length = st.slider("Content Preview Length", min_value=50, max_value=500, value=st.session_state.content_preview_length)

# # Update session state
# if selected_themes:
#     st.session_state.themes = selected_themes
# if custom_text:
#     st.session_state.custom_text = custom_text
# if title_search:
#     st.session_state.title_search = title_search
# st.session_state.n = n
# st.session_state.content_preview_length = content_preview_length

# # Combine selected themes and custom text for input text
# input_text = " ".join(selected_themes) + " " + custom_text

# # Find top n similar texts
# if st.button("Generate Blog Feed"):
#     top_n_content_list = find_top_n_similar_texts(input_text, blog_df, n, content_preview_length, title_search)
#     for item in top_n_content_list:
#         st.markdown(item)
#         st.markdown("---")

# #         st.markdown("---")
