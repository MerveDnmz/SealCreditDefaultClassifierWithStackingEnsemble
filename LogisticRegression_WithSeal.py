import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from ucimlrepo import fetch_ucirepo
from seal import EncryptionParameters, SEALContext, KeyGenerator, Encryptor, Decryptor, CKKSEncoder, Plaintext, Ciphertext, scheme_type, CoeffModulus
import time
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# SEAL ile veri şifreleme fonksiyonları
def seal_encrypt_data(data):
    parms = EncryptionParameters(scheme_type.ckks)
    parms.set_poly_modulus_degree(8192)
    parms.set_coeff_modulus(CoeffModulus.Create(8192, [60, 40, 40, 60]))
    context = SEALContext(parms)

    keygen = KeyGenerator(context)
    public_key = keygen.create_public_key()
    secret_key = keygen.secret_key()

    encryptor = Encryptor(context, public_key)
    encoder = CKKSEncoder(context)
    scale = pow(2.0, 40)

    encrypted_data = []
    for row in data:
        row = np.array(row, dtype=np.float64).flatten()
        plain = encoder.encode(row, scale)
        encrypted_row = encryptor.encrypt(plain)
        encrypted_data.append(encrypted_row)

    return encrypted_data, secret_key, encoder, scale, context

def seal_decrypt_data(encrypted_data, secret_key, encoder, scale, context):
    decrypted_data = []
    decryptor = Decryptor(context, secret_key)
    for encrypted_row in encrypted_data:
        plain = Plaintext()
        decryptor.decrypt(encrypted_row, plain)
        decoded_row = np.array(encoder.decode(plain))
        decrypted_data.append(decoded_row[:23])
    return np.array(decrypted_data)

def build_neural_network(input_shape):
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(input_shape,)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(2, activation='softmax'))  # Assuming binary classification
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def plot_roc_curve(y_test, y_pred_proba_unencrypted, y_pred_proba_encrypted):
    fpr_unencrypted, tpr_unencrypted, _ = roc_curve(y_test, y_pred_proba_unencrypted[:, 1])
    roc_auc_unencrypted = auc(fpr_unencrypted, tpr_unencrypted)

    fpr_encrypted, tpr_encrypted, _ = roc_curve(y_test, y_pred_proba_encrypted[:, 1])
    roc_auc_encrypted = auc(fpr_encrypted, tpr_encrypted)

    plt.figure(figsize=(10, 6))
    plt.plot(fpr_unencrypted, tpr_unencrypted, color='blue', lw=2,
             label=f'Unencrypted Data (AUC = {roc_auc_unencrypted:.2f})')
    plt.plot(fpr_encrypted, tpr_encrypted, color='green', lw=2,
             label=f'Encrypted Data (AUC = {roc_auc_encrypted:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Encrypted and Unencrypted Data')
    plt.legend(loc='lower right')
    plt.grid()
    plt.savefig("ROC_Curve.png")
    plt.show()

def logistic_regression_with_seal():
    default_of_credit_card_clients = fetch_ucirepo(id=350)
    X = default_of_credit_card_clients.data.features
    y = default_of_credit_card_clients.data.targets

    y = y.to_numpy().ravel()
    y = to_categorical(y)  # Convert to one-hot encoding for neural network

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build and train the neural network
    model = build_neural_network(X_train.shape[1])
    start_time = time.time()
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)
    training_time = time.time() - start_time

    # Evaluate the model
    y_pred_proba_unencrypted = model.predict(X_test)
    y_pred_encrypted = model.predict(seal_decrypt_data(seal_encrypt_data(X_test.to_numpy())[0], *seal_encrypt_data(X_test.to_numpy())[1:]))

    # Calculate accuracy
    unencrypted_accuracy = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred_proba_unencrypted, axis=1))
    encrypted_accuracy = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred_encrypted, axis=1))

    print(f"Unencrypted Accuracy: {unencrypted_accuracy:.4f}")
    print(f"Encrypted Accuracy: {encrypted_accuracy:.4f}")

    # Plot ROC curves
    plot_roc_curve(y_test, y_pred_proba_unencrypted, y_pred_encrypted)

if __name__ == "__main__":
    logistic_regression_with_seal()