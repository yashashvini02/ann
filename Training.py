import pandas as pd
import numpy as np
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import pickle
import datetime
#To know about the data
#print(data.info())
#print(data.describe())
#print(data.isnull().sum())
#print(data.describe().sum())
# Load dataset
data = pd.read_csv("Telco-CustomerT-Churn.csv")
# Drop unnecessary columns
if "customerID" in data.columns:
    data = data.drop(columns=["customerID"])
# Convert TotalCharges to numeric and handle missing values
data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
data["TotalCharges"].fillna(0, inplace=True)
label_cols = ["Churn", "gender", "Partner", "Dependents", "PhoneService", "MultipleLines", "PaperlessBilling"]
label_encoders = {}  
for col in label_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le  # Save encoder
# One-hot encode multi-class categorical columns
onehot_cols = [
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract", "PaymentMethod"
]
onehot = OneHotEncoder(drop="first", sparse_output=False)
onehot_df = pd.DataFrame(
    onehot.fit_transform(data[onehot_cols]),
    columns=onehot.get_feature_names_out(onehot_cols),
    index=data.index
)
data = data.drop(columns=onehot_cols)
data = pd.concat([data, onehot_df], axis=1)
y = data["Churn"]
X = data.drop(columns=["Churn"])
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42, stratify=y)
# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_res_np, y_train_res = smote.fit_resample(X_train, y_train)
# Scale features
scaler = StandardScaler()
X_train_res = pd.DataFrame(scaler.fit_transform(X_train_res_np), columns=X.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns)
# Save preprocessing objects
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("onehot.pkl", "wb") as f:
    pickle.dump(onehot, f)
with open("label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)
print("Class distribution after SMOTE:")
print(pd.Series(y_train_res).value_counts())
# Build ANN model
model = Sequential([
    Dense(128, activation="relu", input_shape=(X_train_res.shape[1],)),
    Dropout(0.4),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(32, activation="relu"),
    Dropout(0.2),
    Dense(1, activation="sigmoid")
])
# Compile
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
# Callbacks
early_stop = EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1)
log_dir = "logs/fit_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensor_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
# Class weights
class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train_res), y=y_train_res)
class_weight_dict = dict(enumerate(class_weights))
# Train
history = model.fit(
    X_train_res, y_train_res,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=16,
    callbacks=[tensor_callback, early_stop, reduce_lr],
    class_weight=class_weight_dict,
    verbose=1
)
# Save model
model.save("model.h5")
# Evaluate
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")
