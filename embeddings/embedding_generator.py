"""
Embedding Generator for Game Recommendations
Generates and manages embeddings for games to enable semantic similarity search.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import json
from typing import List, Dict, Tuple, Optional
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

class GameEmbeddingGenerator:
    """Generates and manages embeddings for game recommendations."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding generator.
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.model_name = model_name
        self.model = None
        self.embeddings = None
        self.game_data = None
        self.embeddings_path = Path("embeddings")
        self.embeddings_path.mkdir(exist_ok=True)
        
    def load_model(self):
        """Load the sentence transformer model."""
        try:
            print(f"🔄 Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print("✅ Model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
            
    def load_game_data(self, dataset_path: str):
        """
        Load and preprocess game data from CSV.
        
        Args:
            dataset_path: Path to the video game reviews CSV file
        """
        try:
            print(f"📊 Loading game data from: {dataset_path}")
            df = pd.read_csv(dataset_path)
            
            # Clean and standardize column names
            df.columns = df.columns.str.lower().str.replace(' ', '_')
            
            print(f"✅ Loaded {len(df)} games")
            print(f"📝 Columns: {list(df.columns)}")
            
            # Process the data
            self.game_data = self._process_game_data(df)
            print(f"✅ Processed {len(self.game_data)} games")
            
        except Exception as e:
            print(f"❌ Error loading game data: {e}")
            raise
            
    def _process_game_data(self, df: pd.DataFrame) -> List[Dict]:
        """Process raw game data into a standardized format."""
        games = []
        
        for _, row in df.iterrows():
            # Extract game information
            game = {
                'id': len(games),
                'name': self._extract_text(row, ['title', 'name', 'game', 'game_title']),
                'rating': self._extract_rating(row),
                'price': self._extract_price(row),
                'reviews_count': self._extract_reviews_count(row),
                'description': self._extract_text(row, ['description', 'summary', 'about', 'overview']),
                'genre': self._extract_text(row, ['genre', 'category', 'type', 'genres']),
                'platform': self._extract_text(row, ['platform', 'console', 'system', 'platforms']),
                'year': self._extract_year(row),
                'developer': self._extract_text(row, ['developer', 'publisher', 'studio']),
                'tags': self._extract_tags(row)
            }
            
            # Create text for embedding
            game['embedding_text'] = self._create_embedding_text(game)
            
            games.append(game)
            
        return games
        
    def _extract_text(self, row: pd.Series, columns: List[str]) -> str:
        """Extract text from row using multiple possible column names."""
        for col in columns:
            if col in row and pd.notna(row[col]):
                text = str(row[col]).strip()
                if len(text) > 0:
                    return text
        return ""
        
    def _extract_rating(self, row: pd.Series) -> float:
        """Extract and normalize rating."""
        rating_columns = ['rating', 'score', 'user_rating', 'metascore', 'critic_score']
        for col in rating_columns:
            if col in row and pd.notna(row[col]):
                try:
                    rating = float(row[col])
                    # Normalize to 0-10 scale
                    if rating > 10:
                        rating = rating / 10
                    return round(rating, 1)
                except (ValueError, TypeError):
                    continue
        return 7.0
        
    def _extract_price(self, row: pd.Series) -> float:
        """Extract price information."""
        price_columns = ['price', 'cost', 'price_usd', 'retail_price']
        for col in price_columns:
            if col in row and pd.notna(row[col]):
                try:
                    price = float(row[col])
                    return round(price, 2)
                except (ValueError, TypeError):
                    continue
        return 0.0
        
    def _extract_reviews_count(self, row: pd.Series) -> int:
        """Extract number of reviews."""
        review_columns = ['reviews', 'review_count', 'user_reviews', 'total_reviews']
        for col in review_columns:
            if col in row and pd.notna(row[col]):
                try:
                    return int(row[col])
                except (ValueError, TypeError):
                    continue
        return 100
        
    def _extract_year(self, row: pd.Series) -> int:
        """Extract release year."""
        year_columns = ['year', 'release_year', 'date', 'release_date']
        for col in year_columns:
            if col in row and pd.notna(row[col]):
                try:
                    year = int(row[col])
                    if 1990 <= year <= 2030:
                        return year
                except (ValueError, TypeError):
                    continue
        return 2020
        
    def _extract_tags(self, row: pd.Series) -> List[str]:
        """Extract tags or keywords."""
        tag_columns = ['tags', 'keywords', 'features', 'attributes']
        tags = []
        for col in tag_columns:
            if col in row and pd.notna(row[col]):
                tag_text = str(row[col])
                # Split by common separators
                tag_list = re.split(r'[,;|]', tag_text)
                tags.extend([tag.strip() for tag in tag_list if tag.strip()])
        return list(set(tags))  # Remove duplicates
        
    def _create_embedding_text(self, game: Dict) -> str:
        """Create text representation for embedding generation."""
        parts = []
        
        # Game name
        if game['name']:
            parts.append(game['name'])
            
        # Description
        if game['description']:
            parts.append(game['description'])
            
        # Genre
        if game['genre']:
            parts.append(f"Genre: {game['genre']}")
            
        # Platform
        if game['platform']:
            parts.append(f"Platform: {game['platform']}")
            
        # Tags
        if game['tags']:
            parts.append(f"Features: {', '.join(game['tags'][:5])}")  # Limit to 5 tags
            
        return " | ".join(parts)
        
    def generate_embeddings(self):
        """Generate embeddings for all games."""
        if self.model is None:
            self.load_model()
            
        if self.game_data is None:
            raise ValueError("No game data loaded. Call load_game_data() first.")
            
        print("🔄 Generating embeddings...")
        
        # Extract text for embedding
        texts = [game['embedding_text'] for game in self.game_data]
        
        # Generate embeddings
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        self.embeddings = embeddings
        print(f"✅ Generated {len(embeddings)} embeddings")
        
        # Save embeddings
        self._save_embeddings()
        
    def _save_embeddings(self):
        """Save embeddings and game data to disk."""
        print("💾 Saving embeddings...")
        
        # Save embeddings as numpy array
        embeddings_file = self.embeddings_path / "game_embeddings.npy"
        np.save(embeddings_file, self.embeddings)
        
        # Save game data as JSON
        game_data_file = self.embeddings_path / "game_data.json"
        with open(game_data_file, 'w', encoding='utf-8') as f:
            json.dump(self.game_data, f, indent=2, ensure_ascii=False)
            
        # Save metadata
        metadata = {
            'model_name': self.model_name,
            'num_games': len(self.game_data),
            'embedding_dim': self.embeddings.shape[1],
            'generated_at': pd.Timestamp.now().isoformat()
        }
        
        metadata_file = self.embeddings_path / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        print(f"✅ Saved embeddings to {embeddings_file}")
        print(f"✅ Saved game data to {game_data_file}")
        print(f"✅ Saved metadata to {metadata_file}")
        
    def load_embeddings(self):
        """Load precomputed embeddings from disk."""
        try:
            embeddings_file = self.embeddings_path / "game_embeddings.npy"
            game_data_file = self.embeddings_path / "game_data.json"
            
            if not embeddings_file.exists() or not game_data_file.exists():
                print("❌ Embeddings not found. Run generate_embeddings() first.")
                return False
                
            print("📥 Loading precomputed embeddings...")
            
            self.embeddings = np.load(embeddings_file)
            
            with open(game_data_file, 'r', encoding='utf-8') as f:
                self.game_data = json.load(f)
                
            print(f"✅ Loaded {len(self.game_data)} games with embeddings")
            return True
            
        except Exception as e:
            print(f"❌ Error loading embeddings: {e}")
            return False
            
    def find_similar_games(self, query: str, top_k: int = 10, 
                          filters: Optional[Dict] = None) -> List[Dict]:
        """
        Find similar games based on semantic similarity.
        
        Args:
            query: Search query text
            top_k: Number of similar games to return
            filters: Optional filters (genre, platform, price_range, etc.)
            
        Returns:
            List of similar games with similarity scores
        """
        if self.model is None:
            self.load_model()
            
        if self.embeddings is None:
            if not self.load_embeddings():
                raise ValueError("No embeddings available")
                
        # Generate embedding for query
        query_embedding = self.model.encode([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get top similar games
        top_indices = np.argsort(similarities)[::-1][:top_k * 2]  # Get more for filtering
        
        results = []
        for idx in top_indices:
            game = self.game_data[idx].copy()
            game['similarity_score'] = float(similarities[idx])
            
            # Apply filters if provided
            if filters and not self._apply_filters(game, filters):
                continue
                
            results.append(game)
            
            if len(results) >= top_k:
                break
                
        return results
        
    def _apply_filters(self, game: Dict, filters: Dict) -> bool:
        """Apply filters to a game."""
        # Genre filter
        if 'genre' in filters and filters['genre']:
            if not any(genre.lower() in game['genre'].lower() 
                      for genre in filters['genre']):
                return False
                
        # Platform filter
        if 'platform' in filters and filters['platform']:
            if not any(platform.lower() in game['platform'].lower() 
                      for platform in filters['platform']):
                return False
                
        # Price range filter
        if 'price_range' in filters:
            min_price, max_price = filters['price_range']
            if not (min_price <= game['price'] <= max_price):
                return False
                
        # Rating filter
        if 'min_rating' in filters:
            if game['rating'] < filters['min_rating']:
                return False
                
        # Year range filter
        if 'year_range' in filters:
            min_year, max_year = filters['year_range']
            if not (min_year <= game['year'] <= max_year):
                return False
                
        return True

def main():
    """Main function to generate embeddings."""
    print("🎮 Game Recommendation Bot - Embedding Generator")
    print("=" * 50)
    
    # Check if dataset exists
    dataset_path = Path("dataset/video-game-reviews-and-ratings.csv")
    if not dataset_path.exists():
        print("❌ Dataset not found. Please run download_kaggle_dataset.py first.")
        return
        
    # Initialize generator
    generator = GameEmbeddingGenerator()
    
    try:
        # Load game data
        generator.load_game_data(str(dataset_path))
        
        # Generate embeddings
        generator.generate_embeddings()
        
        print("\n🎉 Embedding generation complete!")
        print("✅ You can now use semantic search for game recommendations")
        
        # Test the system
        print("\n🧪 Testing similarity search...")
        test_queries = [
            "exciting adventure game",
            "relaxing puzzle game",
            "educational game for kids",
            "multiplayer action game"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Query: '{query}'")
            similar_games = generator.find_similar_games(query, top_k=3)
            
            for i, game in enumerate(similar_games, 1):
                print(f"  {i}. {game['name']} (Score: {game['similarity_score']:.3f})")
                
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
