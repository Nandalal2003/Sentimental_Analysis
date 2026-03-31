from src.train import load_data, train_models, save_models
from src.evaluate import evaluate
from src.predict import load_models, predict

# Step 1: Load dataset
dataset = load_data("data/training.csv")

# Step 2: Train
vectoriser, models, X_test, y_test = train_models(dataset)

# Step 3: Evaluate best model (LR)
evaluate(models["LR"], X_test, y_test)

# Step 4: Save models
save_models(vectoriser, models)

# Step 5: Predict
vectoriser, model = load_models()

text = [
    "I love this product!",
    "This is the worst thing ever"
]

print(predict(vectoriser, model, text))
