from sklearn.base import TransformerMixin
from sklearn.calibration import LabelEncoder


class TLabelEncoder(TransformerMixin):
    def __init__(self, *args, **kwargs):
        self.encoder = LabelEncoder(*args, **kwargs)

    def fit(self, X, y=None):
        # Get only categorical values
        self.encoder.fit(X.select_dtypes(include=["object"]))
        return self

    def transform(self, X, y=None):
        return self.encoder.transform(X)
