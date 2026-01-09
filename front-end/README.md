# 🎬📚 Recommendation Chatbot

A modular chatbot interface built with Gradio that provides movie and book recommendations based on user input and mood preferences.

## 🚀 Features

- **Multi-turn Conversations**: Maintains chat history for natural conversations
- **Mood-based Filtering**: Filter recommendations by mood (Happy, Sad, Chill)
- **Dual Content Support**: Recommendations for both movies and books
- **Modular Architecture**: Separate backend (`recommendation.py`) and frontend (`app.py`)
- **Beautiful UI**: Clean, responsive interface with Gradio

## 📁 Project Structure

```
├── recommendation.py    # Backend recommendation logic
├── app.py             # Frontend Gradio chatbot UI
├── requirements.txt   # Python dependencies
└── README.md         # This file
```

## 🛠️ Installation

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the chatbot:**
   ```bash
   python app.py
   ```

3. **Open your browser** and navigate to `http://localhost:7860`

## 💡 Usage

### Basic Usage
- Type your request in the chat (e.g., "I want to watch a good movie")
- The bot will provide 3-5 personalized recommendations
- Continue the conversation for more suggestions or details

### Mood Filtering
- Select a mood from the dropdown (Happy, Sad, Chill, or None)
- Get recommendations tailored to your emotional state
- Mood affects the genre and type of content recommended

### Example Prompts
- **Movies**: "I want to watch a comedy", "Recommend me some action films"
- **Books**: "I want to read a fantasy novel", "Something good to read"
- **With Mood**: Select "Happy" + "I want something uplifting"

## 🔧 Architecture

### Backend (`recommendation.py`)
- `get_recommendations(user_input, mood)`: Core recommendation logic
- `format_recommendations(recommendations)`: Format output for display
- Contains placeholder data for movies and books
- Easily replaceable with real recommendation engine

### Frontend (`app.py`)
- `RecommendationChatbot`: Manages conversation state
- `create_chatbot_interface()`: Builds the Gradio UI
- Handles user interactions and mood filtering
- Maintains chat history across conversations

## 🎯 Customization

### Adding New Content Types
1. Update the `all_recommendations` dictionary in `recommendation.py`
2. Add new content type detection logic in `get_recommendations()`
3. Update the formatting in `format_recommendations()`

### Adding New Moods
1. Add new mood to the dropdown choices in `app.py`
2. Define mood filters in `mood_filters` dictionary in `recommendation.py`
3. Specify which genres/content types match each mood

### Connecting to Real Backend
Replace the placeholder logic in `get_recommendations()` with:
- API calls to recommendation services
- Database queries
- Machine learning model predictions
- External recommendation APIs

## 🚀 Future Enhancements

- **User Profiles**: Save user preferences and history
- **Rating System**: Allow users to rate recommendations
- **Advanced Filtering**: Genre, year, rating, and other filters
- **Recommendation Explanations**: Why specific items were recommended
- **Social Features**: Share recommendations with friends
- **Integration**: Connect to real movie/book databases (IMDb, Goodreads, etc.)

## 🐛 Troubleshooting

### Common Issues
1. **Port already in use**: Change the port in `app.py` (line with `server_port=7860`)
2. **Module not found**: Ensure all dependencies are installed with `pip install -r requirements.txt`
3. **Interface not loading**: Check that Gradio is properly installed and try refreshing the browser

### Getting Help
- Check the console output for error messages
- Ensure Python 3.7+ is being used
- Verify all dependencies are correctly installed

## 📝 License

This project is open source and available under the MIT License.

---

**Ready to get started?** Run `python app.py` and start chatting with your recommendation assistant! 🎉
