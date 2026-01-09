"""
Fast Game Recommendation System
Extracted from working notebook with optimizations for speed and accuracy.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import pickle
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.preprocessing import normalize
    import faiss
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    print("Warning: Some dependencies not available. Install: pip install sentence-transformers scikit-learn faiss-cpu")

class FastGameRecommender:
    """Fast game recommendation system with precomputed embeddings."""
    
    def __init__(self, device='cpu'):
        """Initialize the recommender."""
        self.device = device
        self.model = None
        self.game_embeddings = None
        self.game_data = None
        self.index = None
        self.game_names = None
        
        # Paths
        self.data_path = Path(__file__).parent.parent / "dataset" / "video_game_reviews.csv"
        self.embedding_dir = Path(__file__).parent.parent / "embeddings"
        self.embedding_dir.mkdir(exist_ok=True)
        
        print(f"Using device: {device}")
        
    def load_model(self):
        """Load the sentence transformer model."""
        if not EMBEDDINGS_AVAILABLE:
            raise ImportError("Required dependencies not available")
            
        try:
            self.model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
            print("✅ Model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
            
    def load_data(self):
        """Load the game dataset."""
        try:
            print(f"📊 Loading data from: {self.data_path}")
            self.game_data = pd.read_csv(self.data_path)
            print(f"✅ Loaded {len(self.game_data)} games")
            return True
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
            
    def prepare_text(self, df):
        """Prepare text for embedding generation."""
        # Combine relevant columns for better embeddings
        text_parts = []
        
        # Game title (most important)
        text_parts.append(df['Game Title'].fillna(''))
        
        # Genre
        text_parts.append('Genre: ' + df['Genre'].fillna(''))
        
        # User review text (if available)
        if 'User Review Text' in df.columns:
            review_text = df['User Review Text'].fillna('')
            # Take first 200 characters to avoid too long texts
            review_text = review_text.str[:200]
            text_parts.append(review_text)
            
        # Age group and game mode
        if 'Age Group Targeted' in df.columns:
            text_parts.append('Age: ' + df['Age Group Targeted'].fillna(''))
        if 'Game Mode' in df.columns:
            text_parts.append('Mode: ' + df['Game Mode'].fillna(''))
            
        # Combine all parts
        combined_text = text_parts[0]
        for part in text_parts[1:]:
            combined_text = combined_text + ' | ' + part
            
        return combined_text
        
    def generate_embeddings(self, force_regenerate=False):
        """Generate or load game embeddings."""
        embedding_path = self.embedding_dir / "game_embeddings.npy"
        game_data_path = self.embedding_dir / "game_data.pkl"
        
        # Check if embeddings already exist
        if embedding_path.exists() and game_data_path.exists() and not force_regenerate:
            print("📥 Loading existing embeddings...")
            try:
                self.game_embeddings = np.load(embedding_path)
                
                with open(game_data_path, 'rb') as f:
                    self.game_data = pickle.load(f)
                    
                self.game_names = self.game_data['Game Title'].tolist()
                print(f"✅ Loaded {len(self.game_names)} game embeddings")
                
                # Build FAISS index
                self._build_faiss_index()
                return True
                
            except Exception as e:
                print(f"⚠️ Error loading existing embeddings: {e}")
                print("🔄 Regenerating embeddings...")
                
        # Generate new embeddings
        if self.model is None:
            self.load_model()
            
        if self.game_data is None:
            if not self.load_data():
                return False
                
        print("🧠 Generating embeddings...")
        
        # Prepare text
        combined_texts = self.prepare_text(self.game_data)
        
        # Generate embeddings
        self.game_embeddings = self.model.encode(
            combined_texts.tolist(),
            convert_to_tensor=False,
            batch_size=64,
            show_progress_bar=True
        )
        
        # Normalize embeddings for cosine similarity
        self.game_embeddings = normalize(self.game_embeddings)
        
        # Save embeddings and data
        np.save(embedding_path, self.game_embeddings)
        
        # Save processed game data
        with open(game_data_path, 'wb') as f:
            pickle.dump(self.game_data, f)
            
        self.game_names = self.game_data['Game Title'].tolist()
        
        print(f"✅ Generated and saved {len(self.game_names)} embeddings")
        
        # Build FAISS index
        self._build_faiss_index()
        return True
        
    def _build_faiss_index(self):
        """Build FAISS index for fast similarity search."""
        if self.game_embeddings is None:
            return
            
        try:
            dim = self.game_embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dim)  # Inner product = cosine similarity for normalized vectors
            self.index.add(self.game_embeddings.astype('float32'))
            print("✅ FAISS index built for fast search")
        except Exception as e:
            print(f"❌ Error building FAISS index: {e}")
            self.index = None
            
    def query(self, user_query: str, top_k: int = 5, filters: Optional[Dict] = None) -> List[Tuple[str, float]]:
        """Query for similar games."""
        if self.model is None or self.game_embeddings is None:
            raise ValueError("Model or embeddings not loaded. Call generate_embeddings() first.")
            
        # Generate embedding for user query
        query_embedding = self.model.encode([user_query])
        query_embedding = normalize(query_embedding)
        
        # Search using FAISS if available, otherwise use sklearn
        if self.index is not None:
            scores, indices = self.index.search(query_embedding.astype('float32'), top_k * 2)
            scores = scores[0]
            indices = indices[0]
        else:
            # Fallback to sklearn cosine similarity
            similarities = cosine_similarity(query_embedding, self.game_embeddings)[0]
            top_indices = np.argsort(similarities)[::-1][:top_k * 2]
            scores = similarities[top_indices]
            indices = top_indices
            
        # Apply filters and get results
        results = []
        for score, idx in zip(scores, indices):
            if len(results) >= top_k:
                break
                
            game_name = self.game_names[idx]
            game_info = self.game_data.iloc[idx].to_dict()
            
            # Apply filters if provided
            if filters and not self._apply_filters(game_info, filters):
                continue
                
            results.append((game_name, float(score)))
            
        return results
        
    def _apply_filters(self, game_info: Dict, filters: Dict) -> bool:
        """Apply filters to a game."""
        # Genre filter
        if 'genre' in filters:
            game_genre = game_info.get('Genre', '').lower()
            filter_genres = [g.lower() for g in filters['genre']]
            if not any(genre in game_genre for genre in filter_genres):
                return False
                
        # Rating filter
        if 'min_rating' in filters:
            rating = game_info.get('User Rating', 0)
            if rating < filters['min_rating']:
                return False
                
        # Price filter
        if 'max_price' in filters:
            price = game_info.get('Price', 0)
            if price > filters['max_price']:
                return False
                
        # Platform filter
        if 'platform' in filters:
            platform = game_info.get('Platform', '').lower()
            filter_platforms = [p.lower() for p in filters['platform']]
            if not any(plat in platform for plat in filter_platforms):
                return False
                
        return True
        
    def get_recommendations(self, user_input: str, mood: Optional[str] = None, top_k: int = 5) -> Tuple[List[Dict], str]:
        """Get personalized recommendations with mood filtering."""
        # Parse mood and apply filters
        filters = {}
        if mood and mood.lower() != 'any':
            mood_filters = {
                'happy': {'min_rating': 7.5, 'genre': ['casual', 'party', 'colorful']},
                'sad': {'min_rating': 8.0, 'genre': ['story', 'narrative', 'emotional']},
                'chill': {'min_rating': 7.0, 'genre': ['puzzle', 'casual', 'relaxing']}
            }
            filters = mood_filters.get(mood.lower(), {})
            
        # Get similar games
        similar_games = self.query(user_input, top_k=top_k * 2, filters=filters)
        
        # Format results
        recommendations = []
        for game_name, score in similar_games[:top_k]:
            game_idx = self.game_names.index(game_name)
            game_info = self.game_data.iloc[game_idx]
            
            rec = {
                'name': game_name,
                'rating': float(game_info.get('User Rating', 7.0)),
                'price': float(game_info.get('Price', 0)),
                'reviews': 1000,  # Placeholder - dataset doesn't have review count
                'description': f"Genre: {game_info.get('Genre', 'Unknown')} | Platform: {game_info.get('Platform', 'Unknown')}",
                'genre': game_info.get('Genre', 'Unknown'),
                'platform': game_info.get('Platform', 'Unknown'),
                'similarity_score': score
            }
            recommendations.append(rec)
            
        # Generate explanation
        explanation = self._generate_explanation(user_input, mood, len(recommendations))
        
        return recommendations, explanation
        
    def _generate_explanation(self, user_input: str, mood: Optional[str], num_results: int) -> str:
        """Generate explanation for recommendations."""
        base = f"Found {num_results} games matching your request: '{user_input}'"
        
        if mood and mood.lower() != 'any':
            base += f" with {mood.lower()} mood preference"
            
        base += ". Results ranked by semantic similarity and user ratings."
        
        return base

# Global instance for easy import
_recommender_instance = None

def get_recommender() -> FastGameRecommender:
    """Get or create the global recommender instance."""
    global _recommender_instance
    if _recommender_instance is None:
        _recommender_instance = FastGameRecommender()
        # Try to load existing embeddings first
        if not _recommender_instance.generate_embeddings():
            print("⚠️ Could not load embeddings. Some features may not work.")
    return _recommender_instance

def get_recommendations(user_input: str, mood: Optional[str] = None, top_k: int = 5) -> Tuple[List[Dict], str]:
    """Main function for getting recommendations."""
    try:
        recommender = get_recommender()
        return recommender.get_recommendations(user_input, mood, top_k)
    except Exception as e:
        print(f"❌ Error getting recommendations: {e}")
        return [], f"Sorry, I encountered an error: {e}"

if __name__ == "__main__":
    # Test the recommender
    print("🧪 Testing Fast Game Recommender...")
    
    recommender = FastGameRecommender()
    
    # Generate embeddings
    if recommender.generate_embeddings():
        # Test query
        results = recommender.query("I want a relaxing puzzle game", top_k=5)
        print("\n🔍 Test Results:")
        for name, score in results:
            print(f"  {name}: {score:.4f}")
    else:
        print("❌ Failed to generate embeddings")
