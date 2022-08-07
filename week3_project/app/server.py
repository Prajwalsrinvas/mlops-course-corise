from datetime import datetime

import joblib
from fastapi import FastAPI
from loguru import logger
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

GLOBAL_CONFIG = {
    "model": {
        "featurizer": {
            "sentence_transformer_model": "all-mpnet-base-v2",
            "sentence_transformer_embedding_dim": 768,
        },
        "classifier": {"serialized_model_path": "../data/news_classifier.joblib"},
    },
    "service": {"log_destination": "../data/logs.out"},
}


class PredictRequest(BaseModel):
    source: str
    url: str
    title: str
    description: str


class PredictResponse(BaseModel):
    scores: dict
    label: str


class TransformerFeaturizer(BaseEstimator, TransformerMixin):
    def __init__(self, dim, sentence_transformer_model):
        self.dim = dim
        self.sentence_transformer_model = sentence_transformer_model

    # estimator. Since we don't have to learn anything in the featurizer, this is a no-op
    def fit(self, X, y=None):
        return self

    # transformation: return the encoding of the document as returned by the transformer model
    def transform(self, X, y=None):
        X_t = []
        for doc in X:
            X_t.append(self.sentence_transformer_model.encode(doc))
        return X_t


class TransformerFeaturizer(BaseEstimator, TransformerMixin):
    def __init__(self, dim, sentence_transformer_model):
        self.dim = dim
        self.sentence_transformer_model = sentence_transformer_model
        # you can add any other params to be passed to the constructor here

    # estimator. Since we don't have to learn anything in the featurizer, this is a no-op
    def fit(self, X, y=None):
        return self

    # transformation: return the encoding of the document as returned by the transformer model
    def transform(self, X, y=None):
        """
        Goal: TransformerFeaturizer's transform() method converts the raw text document
        into a feature vector to be passed as input to the classifier.

        Given below is a dummy implementation that always maps it to a zero vector.
        You have to implement this function so it uses computes a document embedding
        of the input document using self.sentence_transformer_model.
        This will be our feature representation of the document
        """

        return self.sentence_transformer_model.encode(
            X, normalize_embeddings=True, batch_size=128
        )


class NewsCategoryClassifier:
    def __init__(self, config: dict) -> None:
        self.config = config
        """
        1. Load the sentence transformer model and initialize the `featurizer` of type `TransformerFeaturizer` (Hint: revisit Week 1 Step 4)
        2. Load the serialized model as defined in GLOBAL_CONFIG['model'] into memory and initialize `model`
        """
        config_model_featurizer = config["model"]["featurizer"]
        featurizer = TransformerFeaturizer(
            dim=config_model_featurizer["sentence_transformer_embedding_dim"],
            sentence_transformer_model=SentenceTransformer(
                config_model_featurizer["sentence_transformer_model"]
            ),
        )

        self.model = joblib.load(
            GLOBAL_CONFIG["model"]["classifier"]["serialized_model_path"]
        )

        self.pipeline = Pipeline(
            [("transformer_featurizer", featurizer), ("classifier", self.model)]
        )

    def predict_proba(self, model_input: dict) -> dict:
        """
        Using the `self.pipeline` constructed during initialization,
        run model inference on a given model input, and return the
        model prediction probability scores across all labels

        Output format:
        {
            "label_1": model_score_label_1,
            "label_2": model_score_label_2
            ...
        }
        """

        predictions = self.pipeline.predict_proba([model_input])
        probability_scores = {
            self.model.classes_[i]: predictions[0][i]
            for i in range(len(self.model.classes_))
        }
        return probability_scores

    def predict_label(self, model_input: dict) -> str:
        """
        Using the `self.pipeline` constructed during initialization,
        run model inference on a given model input, and return the
        model prediction label

        Output format: predicted label for the model input
        """

        predicted_label = self.pipeline.predict([model_input])[0]
        return predicted_label


app = FastAPI()


@app.on_event("startup")
def startup_event():
    """
    2. Initialize the `NewsCategoryClassifier` instance to make predictions online. You should pass any relevant config parameters from `GLOBAL_CONFIG` that are needed by NewsCategoryClassifier
    3. Open an output file to write logs, at the destimation specififed by GLOBAL_CONFIG['service']['log_destination']

    Access to the model instance and log file will be needed in /predict endpoint, make sure you
    store them as global variables
    """
    global classifier
    global logfile
    classifier = NewsCategoryClassifier(GLOBAL_CONFIG)
    logfile = GLOBAL_CONFIG["service"]["log_destination"]
    logger.add(logfile, mode="a")
    logger.info("Setup completed")


@app.on_event("shutdown")
def shutdown_event():
    # clean up
    """
    1. Make sure to flush the log file and close any file pointers to avoid corruption
    2. Any other cleanups
    """
    logger.info("Shutting down application")
    logger.remove()


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    # get model prediction for the input request
    # construct the data to be logged
    # construct response
    """
    1. run model inference and get model predictions for model inputs specified in `request`
    2. Log the following data to the log file (the data should be logged to the file that was opened in `startup_event`, and writes to the path defined in GLOBAL_CONFIG['service']['log_destination'])
    {
        'timestamp': <YYYY:MM:DD HH:MM:SS> format, when the request was received,
        'request': dictionary representation of the input request,
        'prediction': dictionary representation of the response,
        'latency': time it took to serve the request, in millisec
    }
    3. Construct an instance of `PredictResponse` and return
    """
    start_time = datetime.now()
    response = PredictResponse(
        scores=classifier.predict_proba(request.description),
        label=classifier.predict_label(request.description),
    )
    end_time = datetime.now()
    latency = (end_time - start_time).total_seconds() * 1000

    log_data = {
        "timestamp": f"{start_time:%Y:%m:%d, %H:%M:%S}",
        "request": request,
        "prediction": response,
        "latency": latency,
    }
    logger.info(log_data)
    return response


@app.get("/")
def read_root():
    return {"Hello": "World"}
