def seal_encrypt_data(data, context):
    """
    Encrypts the given data using the SEAL library.

    Parameters:
    data : ndarray
        The data to be encrypted.
    context : SEALContext
        The SEAL context containing encryption parameters.

    Returns:
    encrypted_data : list
        A list of encrypted data rows.
    """
    keygen = KeyGenerator(context)
    public_key = keygen.create_public_key()
    encryptor = Encryptor(context, public_key)
    encoder = CKKSEncoder(context)
    scale = pow(2.0, 40)

    encrypted_data = []
    for row in data:
        row = np.array(row, dtype=np.float64).flatten()
        plain = encoder.encode(row, scale)
        encrypted_row = encryptor.encrypt(plain)
        encrypted_data.append(encrypted_row)

    return encrypted_data

def seal_decrypt_data(encrypted_data, secret_key, encoder, context):
    """
    Decrypts the given encrypted data using the SEAL library.

    Parameters:
    encrypted_data : list
        The encrypted data to be decrypted.
    secret_key : SecretKey
        The secret key for decryption.
    encoder : CKKSEncoder
        The encoder used for encoding the data.
    context : SEALContext
        The SEAL context containing encryption parameters.

    Returns:
    decrypted_data : ndarray
        The decrypted data as a NumPy array.
    """
    decrypted_data = []
    decryptor = Decryptor(context, secret_key)
    for encrypted_row in encrypted_data:
        plain = Plaintext()
        decryptor.decrypt(encrypted_row, plain)
        decoded_row = np.array(encoder.decode(plain))
        decrypted_data.append(decoded_row)

    return np.array(decrypted_data)