import streamlit as st
import pandas as pd
import pymongo
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the pickled DataFrame
df = pd.read_pickle('cafes.pkl')

# MongoDB connection
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client['cafe_db']
collection = db['cafe_data']

# Function to get unique localities from MongoDB
def get_localities():
    localities = collection.distinct('locality')
    return localities

# Function to recommend cafes based on café name
def recommend_cafe_by_name(cafe_name):
    unique_cafes = df.drop_duplicates(subset=['name'])
    selected_cafe = unique_cafes[unique_cafes['name'] == cafe_name]
    if selected_cafe.empty:
        st.warning(f"{cafe_name} is not in the list.")
        return pd.DataFrame()  # Return empty DataFrame if not found
    selected_cafe_reviews = selected_cafe['reviews'].values[0]
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(unique_cafes['reviews'])
    cosine_sim = cosine_similarity(tfidf.transform([selected_cafe_reviews]), tfidf_matrix)
    sim_scores = list(enumerate(cosine_sim[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    cafe_indices = [i[0] for i in sim_scores[1:11]] 
    return unique_cafes.iloc[cafe_indices]

# Function to recommend cafes based on user customization
def recommend_cafe_custom(budget=None, locality=None, live_music=None, veg_nonveg=None, bar=None):
    unique_cafes = df.drop_duplicates(subset=['name'])
    conditions = [
        (unique_cafes['price_for_two'] <= budget) if budget is not None and budget > 0 else True,
        (unique_cafes['locality'] == locality) if locality else True,
        (unique_cafes['live_music'] == live_music) if live_music is not None else True,
        (unique_cafes['serves_nonveg'] == veg_nonveg) if veg_nonveg is not None else True,
        (unique_cafes['bar'] == bar) if bar is not None else True,  
    ]

    recommendations = unique_cafes[conditions[0] & conditions[1] & conditions[2] & conditions[3] & conditions[4]]
    
    return recommendations[['name', 'price_for_two', 'live_music', 'serves_nonveg', 'bar', 'cuisines', 'ratings_text', 'url', 'sentiment_positive', 'sentiment_negative']]

# Function to filter reviews based on given keywords
def filter_reviews_by_keywords(reviews, keywords):
    filtered_reviews = []
    for review in reviews:
        if any(keyword in review['description'].lower() for keyword in keywords):
            filtered_reviews.append(review)
    return filtered_reviews

# Function to load words from a file
def load_words_from_file(file_path):
    with open(file_path, 'r') as file:
        words = [line.strip() for line in file if line.strip()]  # Remove empty lines
    return words

# Load positive and negative words from text files
positive_word_counts = load_words_from_file('positive_words.txt')
negative_word_counts = load_words_from_file('negative_words.txt')

# Function to display sentiment analysis results
def display_sentiment_analysis(row):
    sentiments = [row['sentiment_positive'] if not pd.isna(row['sentiment_positive']) else 0,
                  row['sentiment_negative'] if not pd.isna(row['sentiment_negative']) else 0]
    
    labels = ['Positive', 'Negative']
    
    if sentiments[0] == 0 and sentiments[1] == 0:
        st.write("")
    else:
        fig, ax = plt.subplots(figsize=(1, 1))
        ax.pie(sentiments, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#66c2a5', '#fc8d62'], textprops={'fontsize': 5})
        ax.axis('equal')  
        st.pyplot(fig)

        if row['sentiment_positive'] > row['sentiment_negative']:
            st.success("People are generally positive about this place.")
        elif row['sentiment_positive'] < row['sentiment_negative']:
            st.error("People are generally negative about this place.")
        else:
            st.warning("Sentiment about this place is neutral.")

# Function to display reviews
def display_reviews(cafe_name):
    cafe_data = collection.find_one({"name": cafe_name})
    if cafe_data and 'reviews' in cafe_data:
        reviews = cafe_data['reviews']
        
        if reviews:
            positive_reviews = filter_reviews_by_keywords(reviews, positive_word_counts)
            negative_reviews = filter_reviews_by_keywords(reviews, negative_word_counts)

            top_positive_reviews = sorted(positive_reviews, key=lambda x: x['description'].count('good'), reverse=True)[:2]
            top_negative_reviews = sorted(negative_reviews, key=lambda x: x['description'].count('bad'), reverse=True)[:2]
            
            st.write("---")
            st.subheader("Top Positive Reviews")
            for review in top_positive_reviews:
                st.write(f"**{review['author']}**: {review['description']}")
            st.write("---")
            st.subheader("Top Negative Reviews")
            for review in top_negative_reviews:
                st.write(f"**{review['author']}**: {review['description']}")
        else:
            st.write("No reviews available for this café.")
    else:
        st.write("No reviews available for this café.")
def get_star_rating(rating_text):
    # Mapping ratings to star representations
    rating_map = {
        'Excellent': '⭐⭐⭐⭐⭐',
        'Very Good': '⭐⭐⭐⭐',
        'Good': '⭐⭐⭐',
        'Average': '⭐⭐',
        'Poor': '⭐',
        'Not rated': 'No Rating'
    }
    return rating_map.get(rating_text, 'No Rating')

# Unified function to show café recommendations
def show_cafe_recommendations(recommendations):
    if not recommendations.empty:
        for index, row in recommendations.iterrows():
            with st.expander(f"**{row['name']}**"):                
                # Constants for icon paths (Replace these URLs with your icon paths)
                budget_icon = 'https://cdn-icons-png.freepik.com/512/8107/8107808.png'
                live_music_icon = 'https://cdn-icons-png.flaticon.com/512/4507/4507661.png'
                serves_nonveg_icon = 'https://cdn-icons-png.freepik.com/512/4826/4826928.png'
                bar_icon = 'https://cdn-icons-png.flaticon.com/512/2766/2766345.png'

                # Display budget icon and value
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.image(budget_icon, width=100)  # Display icon
                    st.write(f"**Budget for 2:** {row['price_for_two']}")
                
                # Display live music icon and value
                with col2:
                    st.image(live_music_icon, width=100)  # Display icon
                    st.write(f"**Live Music:** {'Yes' if row['live_music'] else 'No'}")
                
                # Display serves non-veg icon and value
                with col3:
                    st.image(serves_nonveg_icon, width=100)  # Display icon
                    st.write(f"**Serves Non-Veg:** {'Yes' if row['serves_nonveg'] else 'No'}")
                
                # Display bar icon and value
                with col4:
                    st.image(bar_icon, width=100)  # Display icon
                    st.write(f"    **Bar:** {'Yes' if row['bar'] else 'No'}")
                st.write("---")
                # Display other details
                st.write(f"**Cuisines:** {row['cuisines']}")
                star_rating = get_star_rating(row['ratings_text'])
                st.write(f"**Rating:** {row['ratings_text']} {star_rating}")
                st.write(f"**URL:** [Visit Café]({row['url']})")
                
                # Display sentiment analysis and reviews
                display_sentiment_analysis(row)
                display_reviews(row['name'])
                st.write("---")
    else:
        st.warning("No recommendations found based on your criteria.")

# Streamlit UI
icon_url = "https://cdn-icons-png.flaticon.com/512/1114/1114350.png"  # Replace with your icon URL

# Use st.markdown to include the icon and title
st.markdown(f"""
    <div style='text-align: center;'>
        <img src='{icon_url}' alt='icon' style='height: 80px;'>
        <h1>Aukàat Séi: Café Recommendation</h1>
    </div>
""", unsafe_allow_html=True)


# Input for café name
selected_cafe_name = st.text_input("Enter a café name:")

# Button to get recommendations based on café name
if st.button("Search Café"):
    recommendations = recommend_cafe_by_name(selected_cafe_name)
    show_cafe_recommendations(recommendations)

# Customization options in the sidebar
st.sidebar.header("Customize Your Recommendations")

localities = get_localities()
selected_locality = st.sidebar.selectbox("Select a locality", localities)
budget = st.sidebar.number_input("Enter your budget for two:", min_value=0, step=50, value=None)
veg_nonveg = st.sidebar.selectbox("Serves Non - veg:", [None, 1, 0], format_func=lambda x: "Yes" if x else "No" if x == 0 else "Any")
live_music = st.sidebar.selectbox("Do you need live music?", [None, 1, 0], format_func=lambda x: "Yes" if x else "No" if x == 0 else "Any")
bar = st.sidebar.selectbox("Do you need a bar?", [None, 1, 0], format_func=lambda x: "Yes" if x else "No" if x == 0 else "Any")

# Button to get recommendations based on customization
if st.sidebar.button("Get Custom Recommendations"):
    recommendations = recommend_cafe_custom(
        budget=budget,
        locality=selected_locality,
        live_music=live_music,
        veg_nonveg=veg_nonveg,
        bar=bar
    )
    show_cafe_recommendations(recommendations)
