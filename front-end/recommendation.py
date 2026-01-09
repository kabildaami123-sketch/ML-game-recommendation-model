"""
Backend recommendation system with placeholder functions.
This module provides recommendation logic that can be easily replaced
with a real recommendation engine later.
"""

import random
from typing import List, Dict, Optional


def get_recommendations(user_input: str, mood: Optional[str] = None) -> List[Dict[str, str]]:
    """
    Generate recommendations based on user input and mood.
    
    Args:
        user_input (str): User's input text
        mood (str, optional): User's mood filter (Happy, Sad, Chill)
    
    Returns:
        List[Dict[str, str]]: List of recommendation dictionaries
    """
    
    # Sample recommendation data
    all_recommendations = {
        "movies": [
            {"title": "The Shawshank Redemption", "genre": "Drama", "year": "1994", "rating": "9.3"},
            {"title": "The Godfather", "genre": "Crime", "year": "1972", "rating": "9.2"},
            {"title": "The Dark Knight", "genre": "Action", "year": "2008", "rating": "9.0"},
            {"title": "Pulp Fiction", "genre": "Crime", "year": "1994", "rating": "8.9"},
            {"title": "Forrest Gump", "genre": "Drama", "year": "1994", "rating": "8.8"},
            {"title": "Inception", "genre": "Sci-Fi", "year": "2010", "rating": "8.8"},
            {"title": "The Matrix", "genre": "Sci-Fi", "year": "1999", "rating": "8.7"},
            {"title": "Goodfellas", "genre": "Crime", "year": "1990", "rating": "8.7"},
            {"title": "The Lord of the Rings: The Fellowship of the Ring", "genre": "Fantasy", "year": "2001", "rating": "8.8"},
            {"title": "Fight Club", "genre": "Drama", "year": "1999", "rating": "8.8"},
        ],
        "books": [
            {"title": "To Kill a Mockingbird", "author": "Harper Lee", "genre": "Fiction", "year": "1960"},
            {"title": "1984", "author": "George Orwell", "genre": "Dystopian", "year": "1949"},
            {"title": "Pride and Prejudice", "author": "Jane Austen", "genre": "Romance", "year": "1813"},
            {"title": "The Great Gatsby", "author": "F. Scott Fitzgerald", "genre": "Fiction", "year": "1925"},
            {"title": "Harry Potter and the Philosopher's Stone", "author": "J.K. Rowling", "genre": "Fantasy", "year": "1997"},
            {"title": "The Catcher in the Rye", "author": "J.D. Salinger", "genre": "Fiction", "year": "1951"},
            {"title": "Lord of the Flies", "author": "William Golding", "genre": "Fiction", "year": "1954"},
            {"title": "The Hobbit", "author": "J.R.R. Tolkien", "genre": "Fantasy", "year": "1937"},
            {"title": "The Chronicles of Narnia", "author": "C.S. Lewis", "genre": "Fantasy", "year": "1950"},
            {"title": "Dune", "author": "Frank Herbert", "genre": "Sci-Fi", "year": "1965"},
        ]
    }
    
    # Mood-based filtering
    mood_filters = {
        "Happy": ["Comedy", "Romance", "Fantasy", "Adventure"],
        "Sad": ["Drama", "Romance", "Fiction"],
        "Chill": ["Fantasy", "Sci-Fi", "Adventure", "Fiction"]
    }
    
    # Determine content type from user input
    content_type = "movies"  # default
    if any(word in user_input.lower() for word in ["book", "books", "novel", "read", "reading"]):
        content_type = "books"
    elif any(word in user_input.lower() for word in ["movie", "movies", "film", "cinema", "watch", "watching"]):
        content_type = "movies"
    
    # Get recommendations based on content type
    recommendations = all_recommendations[content_type].copy()
    
    # Apply mood filter if specified
    if mood and mood in mood_filters:
        if content_type == "movies":
            # For movies, filter by genre
            filtered_recs = [rec for rec in recommendations if rec["genre"] in mood_filters[mood]]
            if filtered_recs:
                recommendations = filtered_recs
        else:
            # For books, filter by genre
            filtered_recs = [rec for rec in recommendations if rec["genre"] in mood_filters[mood]]
            if filtered_recs:
                recommendations = filtered_recs
    
    # Return 3-5 random recommendations
    num_recommendations = random.randint(3, 5)
    selected_recommendations = random.sample(recommendations, min(num_recommendations, len(recommendations)))
    
    return selected_recommendations


def format_recommendations(recommendations: List[Dict[str, str]]) -> str:
    """
    Format recommendations into a readable string.
    
    Args:
        recommendations (List[Dict[str, str]]): List of recommendation dictionaries
    
    Returns:
        str: Formatted recommendation string
    """
    if not recommendations:
        return "Sorry, I couldn't find any recommendations for you right now."
    
    formatted_recs = []
    for i, rec in enumerate(recommendations, 1):
        if "rating" in rec:  # Movie format
            formatted_recs.append(f"{i}. **{rec['title']}** ({rec['year']}) - {rec['genre']} - ⭐ {rec['rating']}")
        else:  # Book format
            formatted_recs.append(f"{i}. **{rec['title']}** by {rec['author']} ({rec['year']}) - {rec['genre']}")
    
    return "\n".join(formatted_recs)


# Example usage and testing
if __name__ == "__main__":
    # Test the recommendation system
    test_inputs = [
        "I want to watch a good movie",
        "Recommend me some books to read",
        "I'm feeling sad, what should I watch?",
        "I want something chill to read"
    ]
    
    for test_input in test_inputs:
        print(f"\nInput: {test_input}")
        recs = get_recommendations(test_input)
        print(format_recommendations(recs))
