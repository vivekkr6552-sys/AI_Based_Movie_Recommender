import streamlit as st
import pandas as pd
import pickle
import os
import requests
from google import genai
from model import load_and_preprocess, compute_similarity, recommend  # Cleaned up imports

# Initialize Gemini Client
# genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
client = genai.Client(api_key=st.secrets["GOOGLE_API_KEY)

st.set_page_config(page_title="AI Based Personalized Movie & Recommender System ", layout="wide")



def display_interactive_recommendations(recommended_movies, movie_ids):
    # CSS polish for hover effects and clean card design
    st.markdown("""
        <style>
        .movie-card {
            transition: transform 0.2s ease-in-out;
            border-radius: 10px;
            padding: 5px;
        }
        .movie-card:hover {
            transform: scale(1.04);
        }
        .movie-title {
            font-weight: 600;
            font-size: 14px;
            text-align: center;
            margin-top: 6px;
            color: #1F2937;
        }
        </style>
    """, unsafe_allow_html=True)

    # 5-Column Grid Layout
    cols = st.columns(5)
    
    for idx, col in enumerate(cols):
        if idx < len(recommended_movies):
            movie_name = recommended_movies[idx]
            m_id = movie_ids[idx]
            poster_url = fetch_poster(m_id, size="w342")

            with col:
                st.markdown('<div class="movie-card">', unsafe_allow_html=True)
                st.image(poster_url, use_container_width=True)
                st.markdown(f'<div class="movie-title">{movie_name}</div>', unsafe_allow_html=True)
                
                # Interactive Expander for extra engagement
                with st.expander("ℹ️ Details"):
                    st.write(f"**ID:** {m_id}")
                    st.link_button("🎬 View Details", f"https://www.themoviedb.org/movie/{m_id}")
                st.markdown('</div>', unsafe_allow_html=True)





def fetch_poster(movie_id):
    tmdb_api_key = os.getenv("TMDB_API_KEY", "dcbe8a5ac320612af892ac05032b61a7")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    }
    
    if tmdb_api_key:
        try:
            # TMDB Images API endpoint call
            url = f"https://api.themoviedb.org/3/movie/{movie_id}/images?api_key={tmdb_api_key}"
            response = requests.get(url, headers=headers, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                posters = data.get('posters', [])
                
                # Agar images API se poster path milta hai
                if posters and len(posters) > 0:
                    file_path = posters[0].get('file_path')
                    return f"https://image.tmdb.org/t/p/w300{file_path}"
            else:
                print(f"TMDB API Status Code Error: {response.status_code}")
                
        except Exception as e:
            print(f"Error fetching poster: {e}")
            
    # Fallback image agar poster na mile
    return "https://placehold.co/500x750/png?text=No+Poster+Available"

def format_currency(amount):
    if amount == 0 or pd.isna(amount):
        return "N/A"
    return f"${amount:,.0f}"

@st.cache_data
def get_data():
    df = load_and_preprocess()
    similarity = compute_similarity(df)
    return df, similarity

df, similarity = get_data()

st.title("🎬 AI Movie Recommender & Assistant")

# Tabs layout: Tab 1 for Recommender, Tab 2 for AI Chatbot
tab1, tab2 = st.tabs(["🍿 Movie Recommender", "🤖 AI Movie Chatbot"])

# ================= TAB 1: RECOMMENDER =================
with tab1:
    st.header("Search Movie Recommendations")
    selected_movie = st.selectbox("Select a movie you like:", df['title'].values)
    
    if st.button("Get Recommendations & Details"):
        results = recommend(selected_movie, df, similarity)
        st.subheader(f"Because you liked **{selected_movie}**, you should watch:")
        
        # Display recommendations with Banner, Cast, Crew, Rating, Budget & Earning
        for item in results:
            col_img, col_details = st.columns([1, 2])
            
            with col_img:
                poster_url = fetch_poster(item['movie_id'])
                st.image(poster_url, use_container_width=True)
                
            with col_details:
                st.markdown(f"### 🎥 {item['title']}")
                st.markdown(f"⭐ **Rating:** {item['rating']} / 10")
                st.markdown(f"🎬 **Director:** {item['director']}")
                st.markdown(f"🎭 **Cast:** {item['cast']}")
                st.markdown(f"💰 **Budget:** {format_currency(item['budget'])}")
                st.markdown(f"💵 **Earning (Revenue):** {format_currency(item['revenue'])}")
                st.markdown(f"📝 **Overview:** {item['overview']}")
            
            st.divider()

# ================= TAB 2: AI CHATBOT =================
with tab2:
    st.header("💬 Chat with AI Movie Assistant")
    st.write("Ask anything about movies, actors, plot summaries, or request custom mood-based suggestions!")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! Main aapka AI Movie Assistant hu. Aap mujhse kisi bhi movie ke suggestions ya details pooch sakte hain!"}
        ]

    # Display prior messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # User input chat box
    if user_prompt := st.chat_input("Ask about movies... (e.g., 'Suggest me top 3 sci-fi movies like Interstellar')"):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.write(user_prompt)

        # Generate Gemini Response
        with st.chat_message("assistant"):
            with st.spinner("AI is thinking..."):
                try:
                    chat_context = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages])
                    prompt = f"You are an expert movie recommender assistant. Answer the user prompt nicely:\n\n{chat_context}"
                    
                    response = client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=prompt
                    )
                    reply = response.text
                    st.write(reply)
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                except Exception as e:
                    st.error(f"Error connecting to AI: {e}")
