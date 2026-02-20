"""PHY-layer spoofing detector using deep neural network."""

import logging
from pathlib import Path
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available, using mock PHY detector")


if TORCH_AVAILABLE:
    class PhySpoofingDetector(nn.Module):
        """Two-stage DNN for PHY-layer spoofing detection (SODA-style)."""
        
        def __init__(
            self,
            input_dim: int = 17,
            hidden_dim: int = 64,
            embedding_dim: int = 32,
            dropout: float = 0.3
        ):
            super().__init__()
            
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.embedding_dim = embedding_dim
            
            # Stage 1: Message classifier (legitimate vs suspicious)
            self.message_classifier = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 2)
            )
            
            # Stage 2: Emitter encoder for consistency checking
            self.emitter_encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, embedding_dim)
            )
            
        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            """
            Forward pass.
            
            Args:
                x: Input features [batch, input_dim]
                
            Returns:
                (classification_logits, emitter_embedding)
            """
            msg_logits = self.message_classifier(x)
            emitter_embedding = self.emitter_encoder(x)
            return msg_logits, emitter_embedding
            
        def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
            """Get spoofing probability."""
            with torch.no_grad():
                logits, _ = self.forward(x)
                probs = F.softmax(logits, dim=1)
                return probs[:, 1]  # P(spoofed)
                
        def compute_consistency_score(
            self,
            current_embedding: torch.Tensor,
            historical_embeddings: torch.Tensor
        ) -> float:
            """
            Compare current transmission to historical RF fingerprint.
            
            Args:
                current_embedding: Embedding of current message
                historical_embeddings: Stack of historical embeddings
                
            Returns:
                Consistency score 0-1 (1 = consistent)
            """
            with torch.no_grad():
                similarity = F.cosine_similarity(
                    current_embedding.unsqueeze(0),
                    historical_embeddings,
                    dim=1
                )
                return float(similarity.mean())


class CalibratedPhyDetector:
    """Calibrated PHY detector with uncertainty estimation."""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        abstain_threshold: float = 0.3
    ):
        self.abstain_threshold = abstain_threshold
        self.model: Optional['PhySpoofingDetector'] = None
        
        if TORCH_AVAILABLE:
            if model_path and Path(model_path).exists():
                self.load_model(model_path)
            else:
                # Create default model
                self.model = PhySpoofingDetector()
                logger.info("Created new PHY detector model")
                
        # History for consistency checking
        self.embedding_history: dict[str, list[np.ndarray]] = {}
        self.max_history = 50
        
    def load_model(self, path: str):
        """Load trained model from file."""
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, cannot load model")
            return
            
        self.model = PhySpoofingDetector()
        self.model.load_state_dict(torch.load(path, map_location='cpu'))
        self.model.eval()
        logger.info(f"Loaded PHY detector from {path}")
        
    def save_model(self, path: str):
        """Save model to file."""
        if self.model is None:
            return
            
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        logger.info(f"Saved PHY detector to {path}")
        
    def predict_with_uncertainty(
        self,
        features: np.ndarray,
        icao24: str
    ) -> dict:
        """
        Predict spoofing with calibrated probability and uncertainty.
        
        Args:
            features: PHY feature vector
            icao24: Aircraft ICAO24 address
            
        Returns:
            Dictionary with prediction, probability, uncertainty, consistency
        """
        if not TORCH_AVAILABLE or self.model is None:
            return {
                "prediction": "unknown",
                "spoof_probability": 0.0,
                "uncertainty": 1.0,
                "abstained": True,
                "consistency_score": 1.0
            }
            
        # Convert to tensor
        x = torch.FloatTensor(features).unsqueeze(0)
        
        with torch.no_grad():
            logits, embedding = self.model(x)
            probs = F.softmax(logits, dim=1)
            
        spoof_prob = float(probs[0, 1])
        uncertainty = float(1 - torch.max(probs))
        
        # Compute consistency with historical embeddings
        embedding_np = embedding.numpy().flatten()
        consistency_score = self._compute_consistency(icao24, embedding_np)
        
        # Store embedding in history
        self._update_history(icao24, embedding_np)
        
        # Determine prediction
        if uncertainty > self.abstain_threshold:
            return {
                "prediction": "uncertain",
                "spoof_probability": spoof_prob,
                "uncertainty": uncertainty,
                "abstained": True,
                "consistency_score": consistency_score
            }
            
        # Combine message classification with consistency
        combined_score = 0.7 * spoof_prob + 0.3 * (1 - consistency_score)
        
        if combined_score > 0.5:
            prediction = "spoofed"
        else:
            prediction = "legitimate"
            
        return {
            "prediction": prediction,
            "spoof_probability": spoof_prob,
            "uncertainty": uncertainty,
            "abstained": False,
            "consistency_score": consistency_score,
            "combined_score": combined_score
        }
        
    def _compute_consistency(self, icao24: str, current_embedding: np.ndarray) -> float:
        """Compute consistency with historical fingerprint."""
        if icao24 not in self.embedding_history:
            return 1.0  # No history, assume consistent
            
        history = self.embedding_history[icao24]
        if len(history) < 3:
            return 1.0
            
        # Average historical embeddings
        historical = np.mean(history, axis=0)
        
        # Cosine similarity
        similarity = np.dot(current_embedding, historical) / (
            np.linalg.norm(current_embedding) * np.linalg.norm(historical) + 1e-10
        )
        
        return float(max(0, similarity))
        
    def _update_history(self, icao24: str, embedding: np.ndarray):
        """Update embedding history for aircraft."""
        if icao24 not in self.embedding_history:
            self.embedding_history[icao24] = []
            
        self.embedding_history[icao24].append(embedding)
        
        if len(self.embedding_history[icao24]) > self.max_history:
            self.embedding_history[icao24].pop(0)


class PhyDetectorTrainer:
    """Training utility for PHY spoofing detector."""
    
    def __init__(
        self,
        model: Optional['PhySpoofingDetector'] = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for training")
            
        self.model = model or PhySpoofingDetector()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()
        
    def train_epoch(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        batch_size: int = 32
    ) -> float:
        """Train for one epoch."""
        self.model.train()
        
        # Shuffle data
        indices = np.random.permutation(len(features))
        features = features[indices]
        labels = labels[indices]
        
        total_loss = 0.0
        num_batches = 0
        
        for i in range(0, len(features), batch_size):
            batch_x = torch.FloatTensor(features[i:i + batch_size])
            batch_y = torch.LongTensor(labels[i:i + batch_size])
            
            self.optimizer.zero_grad()
            
            logits, _ = self.model(batch_x)
            loss = self.criterion(logits, batch_y)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
        return total_loss / num_batches
        
    def evaluate(
        self,
        features: np.ndarray,
        labels: np.ndarray
    ) -> dict:
        """Evaluate model on test data."""
        self.model.eval()
        
        x = torch.FloatTensor(features)
        y = torch.LongTensor(labels)
        
        with torch.no_grad():
            logits, _ = self.model(x)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
        accuracy = float((preds == y).float().mean())
        
        # Per-class metrics
        tp = int(((preds == 1) & (y == 1)).sum())
        fp = int(((preds == 1) & (y == 0)).sum())
        fn = int(((preds == 0) & (y == 1)).sum())
        tn = int(((preds == 0) & (y == 0)).sum())
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn
        }
