import numpy as np
import pandas as pd
from typing import List, Union, Optional
from sklearn.base import BaseEstimator, ClassifierMixin

# Forward reference to avoid circular imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from deepfix_sdk.data.base import InformationRetrievalDataset

from ..data import InformationRetrievalDataset

class IRLookupModel(BaseEstimator, ClassifierMixin):
    """A lookup-based model for Information Retrieval tasks to interface with Deepchecks.
    
    In IR, predictions are usually pre-computed (e.g. scores for query-document pairs).
    This model acts as a wrapper that looks up the pre-computed scores and relevance
    from the IR dataset when Deepchecks evaluates it.
    """
    
    def __init__(self, train_dataset: "InformationRetrievalDataset", test_dataset: Optional["InformationRetrievalDataset"] = None, classes: List[str] = None):
        """Initialize the lookup model.
        
        Args:
            train_dataset: The training dataset.
            test_dataset: The test dataset.
            classes: The class labels for prediction. Defaults to ["0", "1"].
        """
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.classes = classes if classes is not None else ["0", "1"]
        self.classes_ = np.array(self.classes)
            
        lookup_data = []
        datasets = [ds for ds in [self.train_dataset, self.test_dataset] if ds is not None]
        for ds in datasets:
            # predictions and probabilities are aligned with ds.qrels
            qrels = ds.qrels.copy()
            qrels['relevance_pred'] = ds.predictions
            qrels['score_pred'] = ds.probabilities
            lookup_data.append(qrels[['query_id', 'doc_id', 'relevance_pred', 'score_pred']])
            
        retrievals_df = pd.concat(lookup_data, ignore_index=True)
            
        # Ensure query_id and doc_id are strings for consistent lookup
        retrievals_df['query_id'] = retrievals_df['query_id'].astype(str)
        retrievals_df['doc_id'] = retrievals_df['doc_id'].astype(str)
        
        # Drop duplicates just in case (keep first occurrence)
        retrievals_df = retrievals_df.drop_duplicates(subset=['query_id', 'doc_id'], keep='first')
        
        # Set index for O(1) lookup
        self._lookup = retrievals_df.set_index(['query_id', 'doc_id'])

    def fit(self, X, y=None):
        """Mock fit method to satisfy scikit-learn estimator requirements."""
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict relevance for query-document pairs in X."""
        preds = []
        for _, row in X.iterrows():
            q_id = str(row['query_id'])
            e_id = str(row['doc_id'])
            try:
                rel = self._lookup.loc[(q_id, e_id), 'relevance_pred']
                if isinstance(rel, pd.Series):
                    rel = rel.iloc[0]
                preds.append(rel)
            except KeyError:
                raise KeyError(f"Query {q_id} and entity {e_id} not found in lookup table")
        return np.array(preds)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return prediction probabilities for query-document pairs in X."""
        probas = []
        for _, row in X.iterrows():
            q_id = str(row['query_id'])
            e_id = str(row['doc_id'])
            try:
                score = self._lookup.loc[(q_id, e_id), 'score_pred']
                if isinstance(score, pd.Series):
                    score = score.iloc[0]
                probas.append(score)
            except KeyError:
                raise KeyError(f"Query {q_id} and entity {e_id} not found in lookup table")
        return np.array(probas)
        
    def get_params(self, deep=False) -> dict:
        """Override get_params to avoid serializing the datasets."""
        return {"classes": self.classes}
