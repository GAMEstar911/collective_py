import numpy as np
import pandas as pd
import plotly.express as px
from tensorflow import keras
import ml_edu.experiment
import ml_edu.results

# --------------------------------
# Constants
# --------------------------------
DATA_URL = "https://download.mlcc.google.com/mledu-datasets/Rice_Cammeo_Osmancik.csv"

# --------------------------------
# 1. Load Data
# --------------------------------
def load_data(url: str) -> pd.DataFrame:
    """Load the rice dataset from a CSV URL."""
    try:
        df = pd.read_csv(url)
        print("Dataset loaded successfully.")
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset: {e}")


# --------------------------------
# 2. Preprocess Data
# --------------------------------
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Select relevant columns and validate data."""

    required_columns = [
        'Area',
        'Perimeter',
        'Major_Axis_Length',
        'Minor_Axis_Length',
        'Eccentricity',
        'Convex_Area',
        'Extent',
        'Class',
    ]

    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    df = df[required_columns].copy()

    if df.isnull().any().any():
        raise ValueError("Dataset contains missing values.")

    print(f"Preprocessing completed. Rows: {len(df)}")
    return df


# --------------------------------
# 3. Explore Data
# --------------------------------
def explore_data(df: pd.DataFrame) -> None:
    """Print descriptive statistics."""

    shortest = df.Major_Axis_Length.min()
    longest = df.Major_Axis_Length.max()

    min_area = df.Area.min()
    max_area = df.Area.max()

    perimeter = df.Perimeter
    z_score = (perimeter.max() - perimeter.mean()) / perimeter.std()

    print(f"Shortest grain: {shortest:.1f}px")
    print(f"Longest grain: {longest:.1f}px")
    print(f"Smallest area: {min_area}px")
    print(f"Largest area: {max_area}px")
    print(f"Max perimeter Z-score: {z_score:.2f}")


# --------------------------------
# 4. Visualize Data
# --------------------------------
def visualize_data(df: pd.DataFrame) -> None:
    """Generate 2D and 3D plots."""

    pairs = [
        ("Area", "Eccentricity"),
        ("Convex_Area", "Perimeter"),
        ("Major_Axis_Length", "Minor_Axis_Length"),
        ("Perimeter", "Extent"),
        ("Eccentricity", "Major_Axis_Length"),
    ]

    for x, y in pairs:
        px.scatter(df, x=x, y=y, color="Class", title=f"{x} vs {y}").show()

    px.scatter_3d(
        df,
        x="Area",
        y="Perimeter",
        z="Major_Axis_Length",
        color="Class",
    ).show()


# --------------------------------
# 5. Split Data
# --------------------------------
def split_data(df: pd.DataFrame):
    numerical_features = df.select_dtypes('number').columns

    normalized = (df[numerical_features] - df[numerical_features].mean()) / df[numerical_features].std()
    normalized['Class'] = df['Class']
    normalized['Class_Bool'] = (df['Class'] == 'Cammeo').astype(int)

    shuffled = normalized.sample(frac=1, random_state=100)
    n = len(shuffled)

    train = shuffled[:int(0.8 * n)]
    val = shuffled[int(0.8 * n):int(0.9 * n)]
    test = shuffled[int(0.9 * n):]

    label_cols = ['Class', 'Class_Bool']

    return (
        train.drop(columns=label_cols), train['Class_Bool'].to_numpy(),
        val.drop(columns=label_cols), val['Class_Bool'].to_numpy(),
        test.drop(columns=label_cols), test['Class_Bool'].to_numpy(),
    )

# --------------------------------
# 6. Train & Evaluate
# --------------------------------
def create_model(settings, metrics):
    """Create and compile a Keras binary classification model."""

    inputs = {
        feature: keras.Input(
            shape=(1,), name=feature
        )
        for feature in settings.input_features
    }

    # Concatenate inputs
    concatenated = keras.layers.Concatenate()(list(inputs.values()))

    # Hidden layers
    x = keras.layers.Dense(32, activation="relu")(concatenated)
    x = keras.layers.Dense(16, activation="relu")(x)

    # Output layer
    output = keras.layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs=inputs, outputs=output)

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=settings.learning_rate
        ),
        loss="binary_crossentropy",
        metrics=metrics,
    )

    return model
# --------------------------------
# 7. Train & Evaluate
# --------------------------------

def train_model(
    experiment_name: str,
    model: keras.Model,
    dataset: pd.DataFrame,
    labels: np.ndarray,
    settings: ml_edu.experiment.ExperimentSettings,
) -> ml_edu.experiment.Experiment:
    """Train the model and return an Experiment object."""

    # Convert DataFrame into feature dictionary
    features = {
        feature_name: np.array(dataset[feature_name])
        for feature_name in settings.input_features
    }

    history = model.fit(
        x=features,
        y=labels,
        batch_size=settings.batch_size,
        epochs=settings.number_epochs,
        verbose=1,
    )

    return ml_edu.experiment.Experiment(
        name=experiment_name,
        settings=settings,
        model=model,
        epochs=history.epoch,
        metrics_history=pd.DataFrame(history.history),
    )

# --------------------------------
# 8. Train & Evaluate
# --------------------------------
def train_and_evaluate(settings):
    metrics = [
        keras.metrics.BinaryAccuracy(
            name='accuracy',
            threshold=settings.classification_threshold,
        ),
        keras.metrics.Precision(
            name='precision',
            thresholds=settings.classification_threshold,
        ),
        keras.metrics.Recall(
            name='recall',
            thresholds=settings.classification_threshold,
        ),
        keras.metrics.AUC(num_thresholds=100, name='auc'),
    ]

    model = create_model(settings, metrics)

    experiment = train_model(
        experiment_name="experiment",
        model=model,
        dataset=train_features,
        labels=train_labels,
        settings=settings,
    )

    ml_edu.results.plot_experiment_metrics(
        experiment, ['accuracy', 'precision', 'recall']
    )
    ml_edu.results.plot_experiment_metrics(experiment, ['auc'])

    val_metrics = experiment.evaluate(validation_features, validation_labels)
    test_metrics = experiment.evaluate(test_features, test_labels)

    return experiment, val_metrics, test_metrics


# --------------------------------
# 9. Main Pipeline
# --------------------------------
df_raw = load_data(DATA_URL)
df_clean = preprocess_data(df_raw)
explore_data(df_clean)
visualize_data(df_clean)

(
    train_features, train_labels,
    validation_features, validation_labels,
    test_features, test_labels
) = split_data(df_clean)

input_features = [
    'Eccentricity',
    'Major_Axis_Length',
    'Area',
]

settings = ml_edu.experiment.ExperimentSettings(
    learning_rate=0.001,
    number_epochs=60,
    batch_size=100,
    classification_threshold=0.35,
    input_features=input_features,
)

experiment, val_metrics, test_metrics = train_and_evaluate(settings)
