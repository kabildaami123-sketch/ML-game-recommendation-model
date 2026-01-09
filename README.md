# 🎮 Game Recommendation Bot

An intelligent game recommendation system that uses **embeddings + metadata filters** to provide personalized game suggestions based on user input and mood.

## 🌟 Features

- ** Semantic Search**: Uses sentence transformers for intelligent game matching
- ** Metadata Filtering**: Filters by genre, platform, price, rating, and more
- ** Mood-Based Recommendations**: Considers user mood (Happy, Sad, Chill)
- ** Real Data**: Based on actual video game reviews from Kaggle
- ** Beautiful UI**: Gradio chatbot interface with modern design
- ** Fast Performance**: Precomputed embeddings for instant recommendations

##  Project Structure

```
bot/
├── 📂 dataset/                          # Kaggle video game reviews dataset
│   └── video-game-reviews-and-ratings.csv
├── 📂 embeddings/                       # Precomputed game embeddings
│   ├── embedding_generator.py          # Generates embeddings from dataset
│   ├── game_embeddings.npy            # Precomputed embeddings
│   ├── game_data.json                 # Processed game data
│   └── metadata.json                  # Embedding metadata
├── 📂 front-end/                       # Gradio chatbot interface
│   ├── app.py                         # Main chatbot UI
│   ├── recommendation.py              # Enhanced recommendation system
│   └── requirements.txt               # Python dependencies
├── 📂 back-end/                       # Data processing (legacy)
│   ├── data_processor.py              # Basic data processor
│   └── download_dataset.py            # Dataset downloader
├── 📄 recommendation_engine.py        # Advanced recommendation engine
├── 📄 download_kaggle_dataset.py      # Dataset download script
├── 📄 setup_game_bot.py              # Complete setup script
└── 📄 README.md                       # This file
```

##  Quick Start

### 1. Install Dependencies

```bash
cd front-end
pip install -r requirements.txt
```

### 2. Complete Setup (Recommended)

Run the automated setup script:

```bash
python setup_game_bot.py
```

This will:
- Download the Kaggle dataset
- Generate embeddings for semantic search
- Test the entire system
- Verify everything is working

### 3. Manual Setup (Alternative)

If you prefer manual setup:

```bash
# Download dataset
python download_kaggle_dataset.py

# Generate embeddings
python embeddings/embedding_generator.py

# Test the system
python recommendation_engine.py
```

### 4. Start the Bot

```bash
cd front-end
python app.py
```

The bot will be available at `http://localhost:7862`

##  How It Works

### 1. **Data Processing**
- Downloads video game reviews from Kaggle
- Processes and cleans the data
- Extracts game metadata (name, rating, price, genre, etc.)

### 2. **Embedding Generation**
- Uses sentence transformers to create semantic embeddings
- Combines game name, description, genre, and features
- Saves precomputed embeddings for fast retrieval

### 3. **Recommendation Engine**
- Parses user intent from natural language
- Applies mood-based and metadata filters
- Uses semantic similarity to find relevant games
- Ranks results by similarity score and user ratings

### 4. **User Interface**
- Beautiful Gradio chatbot interface
- Mood selector for personalized recommendations
- Real-time conversation with the bot

##  Usage Examples

### Basic Queries
```
"I'm bored and want something exciting"
"I want to learn Spanish through games"
"Looking for relaxing puzzle games"
"I need educational games for my 8-year-old"
```

### Mood-Based Recommendations
- **Happy**: Cheerful, colorful, fun games
- **Sad**: Emotional, story-driven, thoughtful games  
- **Chill**: Relaxing, peaceful, casual games

### Advanced Filtering
The system automatically applies filters based on:
- **Genre**: Adventure, puzzle, educational, etc.
- **Platform**: PC, console, mobile, etc.
- **Price Range**: Free, budget, premium
- **Rating**: Minimum quality threshold
- **Year**: Recent vs. classic games

##  Customization

### Adding New Datasets
1. Place CSV files in the `dataset/` folder
2. Update the data processing logic in `embedding_generator.py`
3. Regenerate embeddings with `python embeddings/embedding_generator.py`

### Modifying Recommendation Logic
Edit `recommendation_engine.py` to:
- Add new intent categories
- Modify mood filters
- Adjust similarity thresholds
- Add custom metadata filters

### Customizing the UI
Edit `front-end/app.py` to:
- Change the color scheme
- Add new mood options
- Modify the chat interface
- Add new features

##  Performance

- **Embedding Generation**: ~2-5 minutes (one-time setup)
- **Recommendation Speed**: <100ms per query
- **Memory Usage**: ~200MB for embeddings
- **Dataset Size**: ~10,000+ games

##  Technical Details

### Dependencies
- **gradio**: Web UI framework
- **sentence-transformers**: Semantic embeddings
- **scikit-learn**: Similarity calculations
- **pandas**: Data processing
- **kagglehub**: Dataset download

### Embedding Model
- **Model**: `all-MiniLM-L6-v2`
- **Dimensions**: 384
- **Language**: English
- **Speed**: Fast inference

### Similarity Search
- **Algorithm**: Cosine similarity
- **Filtering**: Metadata-based pre-filtering
- **Ranking**: Combined similarity + rating score

##  Troubleshooting

### Common Issues

**"No embeddings found"**
```bash
python embeddings/embedding_generator.py
```

**"Dataset not found"**
```bash
python download_kaggle_dataset.py
```

**"Import errors"**
```bash
pip install -r front-end/requirements.txt
```

**"Slow recommendations"**
- Ensure embeddings are precomputed
- Check if `embeddings/` folder exists with `.npy` files

### Performance Issues
- First run may be slow (downloading models)
- Subsequent runs use cached embeddings
- Consider reducing dataset size for faster processing

##  Deployment

### Local Development
```bash
cd front-end
python app.py
```

### Cloud Deployment
The Gradio app can be deployed to:
- **Hugging Face Spaces**
- **Google Colab**
- **AWS/GCP/Azure**
- **Docker containers**

### Production Considerations
- Use GPU for faster embedding generation
- Implement caching for frequently requested games
- Add user preference learning
- Monitor recommendation quality

##  Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

##  License

This project is open source and available under the MIT License.

##  Acknowledgments

- **Kaggle**: For the video game reviews dataset
- **Hugging Face**: For the sentence transformer models
- **Gradio**: For the beautiful UI framework
- **Sentence Transformers**: For semantic similarity capabilities

---

**Happy Gaming! **
