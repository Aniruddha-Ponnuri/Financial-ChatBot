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

        self.logger.info("Initializing Reward Model")

        # Get configuration
        self.embed_model_name = config.get_rl_config(
            "reward_model.embedding_model", "all-MiniLM-L6-v2"
        )
        self.min_samples = config.get_rl_config("reward_model.min_samples_for_training", 20)

        self.logger.info(f"Configuration - Embedding model: {self.embed_model_name}")
        self.logger.info(f"Configuration - Min samples for training: {self.min_samples}")

        # Model directory setup
        self.model_dir = os.path.join(os.path.dirname(__file__), "model_data")
        os.makedirs(self.model_dir, exist_ok=True)
        self.logger.info(f"Model directory: {self.model_dir}")

        self.clf_path = os.path.join(self.model_dir, "reward_classifier.pkl")
        self.logger.info(f"Classifier path: {self.clf_path}")

        # Initialize models
        self.embed_model = None
        self.clf = None
        self.training_count = 0

        self._initialize_models()
        self.logger.info("Reward Model initialization complete")

    def _initialize_models(self):
        """Initialize or load sentence transformer and classifier"""
        try:
            self.logger.info("Starting model initialization")

            # Import here to avoid issues if not installed yet
            from sentence_transformers import SentenceTransformer
            from sklearn.linear_model import SGDClassifier

            self.logger.info(f"Loading sentence transformer model: {self.embed_model_name}")
            self.embed_model = SentenceTransformer(self.embed_model_name)
            self.logger.info("Sentence transformer model loaded successfully")

            # Try to load existing classifier
            if os.path.exists(self.clf_path):
                self.logger.info(f"Found existing classifier at {self.clf_path}")
                with open(self.clf_path, "rb") as f:
                    saved_data = pickle.load(f)
                    self.clf = saved_data["classifier"]
                    self.training_count = saved_data.get("training_count", 0)
                self.logger.info(
                    f"Loaded existing reward model (trained on {self.training_count} samples)"
                )
                self.logger.info(
                    f"Classifier classes: {self.clf.classes_ if hasattr(self.clf, 'classes_') else 'Not yet trained'}"
                )
            else:
                self.logger.info("No existing reward model found")
                self.logger.info("Will create new classifier on first training")

        except ImportError as e:
            self.logger.error(f"Required packages not installed: {e}")
            self.logger.error("Please install: pip install sentence-transformers scikit-learn")
            raise
        except Exception as e:
            self.logger.error(f"Error initializing reward model: {e}")
            self.logger.error(f"Error type: {type(e).__name__}")
            raise

    def _embed_texts(self, texts):
        """Convert texts to embeddings"""
        if self.embed_model is None:
            self.logger.error("Embedding model not initialized")
            raise RuntimeError("Embedding model not initialized")

        self.logger.info(f"Generating embeddings for {len(texts)} texts")
        embeddings = self.embed_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        self.logger.info(f"Embeddings generated - shape: {embeddings.shape}")
        return embeddings

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
            self.logger.warning("No texts provided for scoring")
            return []

        self.logger.info(f"Predicting scores for {len(texts)} candidates")

        # If no classifier trained yet, return uniform scores
        if self.clf is None:
            self.logger.warning(
                "No trained reward model available - returning uniform scores (0.5)"
            )
            return [0.5] * len(texts)

        try:
            self.logger.info("Generating embeddings for score prediction")
            X = self._embed_texts(texts)

            self.logger.info("Predicting probabilities with classifier")
            # Get probability predictions
            probs = self.clf.predict_proba(X)
            self.logger.info(f"Probabilities calculated - shape: {probs.shape}")

            # Get index for positive class (1)
            if 1 in self.clf.classes_:
                class_index = list(self.clf.classes_).index(1)
                scores = probs[:, class_index].tolist()
                self.logger.info(f"Using positive class (1) at index {class_index}")
            else:
                # Fallback if only one class seen
                self.logger.warning(
                    "Classifier has not seen both classes yet - returning uniform scores"
                )
                scores = [0.5] * len(texts)

            self.logger.info(f"Score prediction complete - scores: {[f'{s:.4f}' for s in scores]}")
            return scores

        except Exception as e:
            self.logger.error(f"Error predicting scores: {e}")
            self.logger.error(f"Error type: {type(e).__name__}")
            self.logger.warning("Returning uniform scores due to error")
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
            self.logger.warning("No data provided for model update - skipping")
            return

        if len(texts) != len(labels):
            self.logger.error(
                f"Mismatch between texts ({len(texts)}) and labels ({len(labels)}) length"
            )
            return

        try:
            self.logger.info("=" * 60)
            self.logger.info("Starting reward model update")

            from sklearn.linear_model import SGDClassifier

            self.logger.info("Generating embeddings for training data")
            X = self._embed_texts(texts)
            y = np.array(labels)

            positive_count = sum(y)
            negative_count = len(y) - positive_count

            self.logger.info(f"Updating reward model with {len(texts)} samples")
            self.logger.info(
                f"Label distribution - Positive: {positive_count}, Negative: {negative_count}"
            )
            self.logger.info(f"Positive ratio: {positive_count / len(y) * 100:.1f}%")

            if self.clf is None:
                self.logger.info("No existing classifier - creating new SGDClassifier")
                # Initialize classifier with both classes
                self.clf = SGDClassifier(
                    loss="log_loss",  # For probability estimates
                    max_iter=1000,
                    tol=1e-3,
                    random_state=42,
                )
                # First fit requires classes parameter
                self.logger.info("Performing initial fit with classes [0, 1]")
                self.clf.partial_fit(X, y, classes=np.array([0, 1]))
                self.logger.info("New reward classifier created and trained")
            else:
                # Incremental update
                self.logger.info("Existing classifier found - performing incremental update")
                self.clf.partial_fit(X, y)
                self.logger.info("Incremental update completed")

            self.training_count += len(texts)
            self.logger.info(f"Total training samples: {self.training_count}")

            # Save the updated model
            self._save_model()

            self.logger.info("Reward model update completed successfully")
            self.logger.info("=" * 60)

        except Exception as e:
            self.logger.error(f"Error updating reward model: {e}")
            self.logger.error(f"Error type: {type(e).__name__}")
            self.logger.error("=" * 60)
            raise

    def _save_model(self):
        """Persist the classifier to disk"""
        try:
            self.logger.info(f"Saving reward model to {self.clf_path}")
            saved_data = {
                "classifier": self.clf,
                "training_count": self.training_count,
            }
            with open(self.clf_path, "wb") as f:
                pickle.dump(saved_data, f)
            self.logger.info(
                f"Reward model saved successfully (total training samples: {self.training_count})"
            )
        except Exception as e:
            self.logger.error(f"Error saving reward model: {e}")
            self.logger.error(f"Error type: {type(e).__name__}")

    def is_ready(self):
        """Check if model is ready for inference"""
        return self.clf is not None

    def get_training_count(self):
        """Get number of samples the model has been trained on"""
        return self.training_count
