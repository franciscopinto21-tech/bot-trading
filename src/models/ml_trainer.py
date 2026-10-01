"""
Entrenador de Modelos ML - Gestiona el ciclo de vida de los modelos.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import joblib
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    model_id: str
    model_type: str
    symbol: str
    train_date: datetime
    metrics: Dict[str, float]
    features_used: List[str]
    data_range: Tuple[datetime, datetime]
    version: int


class MLTrainer:
    """
    Gestiona el entrenamiento, validación y reentrenamiento de modelos ML.
    """

    MODELS_DIR = Path("models")

    def __init__(
        self,
        model_type: str = "xgboost",
        retrain_interval_hours: int = 24,
        model_dir: Optional[str] = None,
    ):
        self.model_type = model_type
        self.retrain_interval_hours = retrain_interval_hours
        
        if model_dir:
            self.MODELS_DIR = Path(model_dir)
        
        self.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        
        self.models: Dict[str, Any] = {}
        self.metadata: Dict[str, ModelMetadata] = {}
        self._load_existing_models()

    def _load_existing_models(self):
        """Carga modelos existentes desde disco."""
        if not self.MODELS_DIR.exists():
            return

        for model_file in self.MODELS_DIR.glob("*.joblib"):
            try:
                model_id = model_file.stem
                data = joblib.load(model_file)
                if "model" in data and "metadata" in data:
                    self.models[model_id] = data["model"]
                    self.metadata[model_id] = data["metadata"]
                    logger.info(f"Modelo cargado: {model_id}")
            except Exception as e:
                logger.error(f"Error cargando modelo {model_file}: {e}")

    def train(
        self,
        symbol: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        data_start: Optional[datetime] = None,
        data_end: Optional[datetime] = None,
    ) -> ModelMetadata:
        """
        Entrena un nuevo modelo para un símbolo.
        """
        from xgboost import XGBClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler

        model_id = f"{symbol.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Construir modelo según tipo
        if self.model_type == "xgboost":
            model = XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=-1,
                eval_metric="logloss",
            )
        elif self.model_type == "random_forest":
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=8,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1,
            )
        else:
            raise ValueError(f"Tipo de modelo no soportado: {self.model_type}")

        # Escalar datos
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # Entrenar
        eval_set = None
        if X_val is not None and y_val is not None:
            X_val_scaled = scaler.transform(X_val)
            eval_set = [(X_train_scaled, y_train), (X_val_scaled, y_val)]

        model.fit(
            X_train_scaled,
            y_train,
            eval_set=eval_set,
            verbose=False,
        )

        # Calcular métricas
        train_pred = model.predict(X_train_scaled)
        metrics = self._calculate_metrics(y_train, train_pred)

        if X_val is not None and y_val is not None:
            val_pred = model.predict(X_val_scaled)
            val_metrics = self._calculate_metrics(y_val, val_pred)
            metrics.update({f"val_{k}": v for k, v in val_metrics.items()})

        # Crear metadata
        metadata = ModelMetadata(
            model_id=model_id,
            model_type=self.model_type,
            symbol=symbol,
            train_date=datetime.now(),
            metrics=metrics,
            features_used=feature_names or [],
            data_range=(
                data_start or datetime.now() - timedelta(days=365),
                data_end or datetime.now(),
            ),
            version=len([m for m in self.metadata.values() if m.symbol == symbol]) + 1,
        )

        # Guardar modelo y metadata
        model_data = {
            "model": model,
            "scaler": scaler,
            "metadata": metadata,
        }

        model_path = self.MODELS_DIR / f"{model_id}.joblib"
        joblib.dump(model_data, model_path)

        self.models[model_id] = model
        self.metadata[model_id] = metadata

        logger.info(
            f"Modelo entrenado: {model_id} | "
            f"Accuracy: {metrics.get('accuracy', 0):.2%} | "
            f"F1: {metrics.get('f1', 0):.2%}"
        )

        return metadata

    def predict(
        self, symbol: str, features: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predice usando el mejor modelo disponible para un símbolo.
        Returns: (predicciones, probabilidades)
        """
        model_id = self._get_best_model_id(symbol)
        if not model_id:
            raise ValueError(f"No hay modelo entrenado para {symbol}")

        model_data = joblib.load(self.MODELS_DIR / f"{model_id}.joblib")
        model = model_data["model"]
        scaler = model_data["scaler"]

        features_scaled = scaler.transform(features)
        predictions = model.predict(features_scaled)
        probabilities = model.predict_proba(features_scaled)

        return predictions, probabilities

    def needs_retraining(self, symbol: str) -> bool:
        """Verifica si un modelo necesita reentrenamiento."""
        model_id = self._get_best_model_id(symbol)
        if not model_id:
            return True

        metadata = self.metadata.get(model_id)
        if not metadata:
            return True

        time_since_train = datetime.now() - metadata.train_date
        return time_since_train > timedelta(hours=self.retrain_interval_hours)

    def _get_best_model_id(self, symbol: str) -> Optional[str]:
        """Obtiene el mejor modelo disponible para un símbolo."""
        symbol_models = [
            (mid, meta)
            for mid, meta in self.metadata.items()
            if meta.symbol == symbol
        ]

        if not symbol_models:
            return None

        # Ordenar por F1 score y fecha
        best = max(
            symbol_models,
            key=lambda x: (
                x[1].metrics.get("val_f1", 0),
                x[1].train_date.timestamp(),
            ),
        )

        return best[0]

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calcula métricas de rendimiento."""
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            confusion_matrix,
        )

        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
        }

        # Matriz de confusión
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics["true_positives"] = int(tp)
        metrics["false_positives"] = int(fp)
        metrics["true_negatives"] = int(tn)
        metrics["false_negatives"] = int(fn)

        # Especificidad
        metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0

        return metrics

    def get_model_info(self, symbol: str) -> Optional[Dict]:
        """Obtiene información del modelo para un símbolo."""
        model_id = self._get_best_model_id(symbol)
        if not model_id:
            return None

        metadata = self.metadata[model_id]
        return {
            "model_id": metadata.model_id,
            "model_type": metadata.model_type,
            "train_date": metadata.train_date.isoformat(),
            "metrics": metadata.metrics,
            "features_used": metadata.features_used[:10],  # Top 10 features
            "data_range": {
                "start": metadata.data_range[0].isoformat(),
                "end": metadata.data_range[1].isoformat(),
            },
            "version": metadata.version,
        }

    def list_available_models(self) -> List[Dict]:
        """Lista todos los modelos disponibles."""
        return [
            {
                "symbol": meta.symbol,
                "model_id": meta.model_id,
                "model_type": meta.model_type,
                "train_date": meta.train_date.isoformat(),
                "f1_score": meta.metrics.get("f1", 0),
                "version": meta.version,
            }
            for meta in self.metadata.values()
        ]

    def delete_old_models(self, keep_last_n: int = 5):
        """Elimina modelos antiguos, manteniendo los últimos N por símbolo."""
        from collections import defaultdict

        symbol_models = defaultdict(list)
        for mid, meta in self.metadata.items():
            symbol_models[meta.symbol].append((mid, meta.train_date))

        for symbol, models in symbol_models.items():
            models.sort(key=lambda x: x[1], reverse=True)
            
            if len(models) > keep_last_n:
                for mid, _ in models[keep_last_n:]:
                    model_path = self.MODELS_DIR / f"{mid}.joblib"
                    if model_path.exists():
                        model_path.unlink()
                        logger.info(f"Modelo antiguo eliminado: {mid}")
                    
                    del self.models[mid]
                    del self.metadata[mid]
