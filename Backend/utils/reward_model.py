"""
Reward Model for Reinforcement Learning
Uses sentence embeddings and SGD classifier for scoring candidate responses
"""
import os
import pickle
import numpy as np
from .logger import CustomLogger


class RewardModel:
    """
    Reward model that learns from human feedback to score responses.
    Uses sentence transformers for embeddings and SGDClassifier for incremental learning.
    """
    
    def __init__(self, config, logger: CustomLogger):
        self.logger = logger
        self.config = config
        
        # Get configuration
        self.embed_model_name = config.get_rl_config('reward_model.embedding_model', 'all-MiniLM-L6-v2')
        self.min_samples = config.get_rl_config('reward_model.min_samples_for_training', 20)
        
        # Model directory setup
        self.model_dir = os.path.join(os.path.dirname(__file__), "model_data")
        os.makedirs(self.model_dir, exist_ok=True)
        
        self.clf_path = os.path.join(self.model_dir, "reward_classifier.pkl")
        
        # Initialize models
        self.embed_model = None
        self.clf = None
        self.training_count = 0
        
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize or load sentence transformer and classifier"""
        try:
            # Import here to avoid issues if not installed yet
            from sentence_transformers import SentenceTransformer
            from sklearn.linear_model import SGDClassifier
            
            self.logger.info(f"Loading sentence transformer model: {self.embed_model_name}")
            self.embed_model = SentenceTransformer(self.embed_model_name)
            
            # Try to load existing classifier
            if os.path.exists(self.clf_path):
                with open(self.clf_path, 'rb') as f:
                    saved_data = pickle.load(f)
                    self.clf = saved_data['classifier']
                    self.training_count = saved_data.get('training_count', 0)
                self.logger.info(f"Loaded existing reward model (trained on {self.training_count} samples)")
            else:
                self.logger.info("No existing reward model found, will create new one on first training")
                
        except ImportError as e:
            self.logger.error(f"Required packages not installed: {e}")
            self.logger.error("Please install: pip install sentence-transformers scikit-learn")
            raise
        except Exception as e:
            self.logger.error(f"Error initializing reward model: {e}")
            raise
    
    def _embed_texts(self, texts):
        """Convert texts to embeddings"""
        if self.embed_model is None:
            raise RuntimeError("Embedding model not initialized")
        
        return self.embed_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    
    def predict_scores(self, texts):
        """
        Predict reward scores for given texts.
        Returns probability of positive rating (higher is better).
        
        Args:
            texts: List of strings to score (format: "Q: {question}\\nA: {answer}")
        
        Returns:
            List of scores between 0 and 1
        """
        if not texts:
            return []
        
        # If no classifier trained yet, return uniform scores
        if self.clf is None:
            self.logger.warning("No trained reward model available, returning uniform scores")
            return [0.5] * len(texts)
        
        try:
            X = self._embed_texts(texts)
            
            # Get probability predictions
            probs = self.clf.predict_proba(X)
            
            # Get index for positive class (1)
            if 1 in self.clf.classes_:
                class_index = list(self.clf.classes_).index(1)
                scores = probs[:, class_index].tolist()
            else:
                # Fallback if only one class seen
                self.logger.warning("Classifier has not seen both classes yet")
                scores = [0.5] * len(texts)
            
            self.logger.info(f"Predicted scores for {len(texts)} candidates: {scores}")
            return scores
            
        except Exception as e:
            self.logger.error(f"Error predicting scores: {e}")
            return [0.5] * len(texts)
    
    def update(self, texts, labels):
        """
        Update the reward model with new feedback data.
        Uses partial_fit for incremental learning.
        
        Args:
            texts: List of strings (format: "Q: {question}\\nA: {answer}")
            labels: List of ratings (0 for negative, 1 for positive)
        """
        if not texts or not labels:
            self.logger.warning("No data provided for model update")
            return
        
        if len(texts) != len(labels):
            self.logger.error("Mismatch between texts and labels length")
            return
        
        try:
            from sklearn.linear_model import SGDClassifier
            
            X = self._embed_texts(texts)
            y = np.array(labels)
            
            self.logger.info(f"Updating reward model with {len(texts)} new samples")
            self.logger.info(f"Label distribution - Positive: {sum(y)}, Negative: {len(y) - sum(y)}")
            
            if self.clf is None:
                # Initialize classifier with both classes
                self.clf = SGDClassifier(
                    loss='log_loss',  # For probability estimates
                    max_iter=1000,
                    tol=1e-3,
                    random_state=42
                )
                # First fit requires classes parameter
                self.clf.partial_fit(X, y, classes=np.array([0, 1]))
                self.logger.info("Created new reward classifier")
            else:
                # Incremental update
                self.clf.partial_fit(X, y)
                self.logger.info("Updated existing reward classifier")
            
            self.training_count += len(texts)
            
            # Save the updated model
            self._save_model()
            
        except Exception as e:
            self.logger.error(f"Error updating reward model: {e}")
            raise
    
    def _save_model(self):
        """Persist the classifier to disk"""
        try:
            saved_data = {
                'classifier': self.clf,
                'training_count': self.training_count
            }
            with open(self.clf_path, 'wb') as f:
                pickle.dump(saved_data, f)
            self.logger.info(f"Reward model saved (total training samples: {self.training_count})")
        except Exception as e:
            self.logger.error(f"Error saving reward model: {e}")
    
    def is_ready(self):
        """Check if model is ready for inference"""
        return self.clf is not None
    
    def get_training_count(self):
        """Get number of samples the model has been trained on"""
        return self.training_count
