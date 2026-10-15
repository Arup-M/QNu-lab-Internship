# -*- coding: utf-8 -*-
# tenseal_utils.py
import tenseal as ts
import numpy as np
import logging  
import os

# TenSEAL Manager for CKKS and BFV schemes
# This class manages the creation of TenSEAL contexts, encryption, decryption, and serialization
# It supports both CKKS and BFV schemes, allowing for flexible use in homomorphic encryption
# The class provides methods to encrypt and decrypt vectors, rotate encrypted vectors (only for CKKS),
# and serialize the context for saving/loading. It also includes error handling for unsupported operations.
class TenSealManager:
    def __init__(self, scheme="ckks", scale=2**21, poly_modulus_degree=8192, plain_modulus=1032193, verbose=False):
        """
        Create a TenSEAL context for either CKKS or BFV scheme.
        :param scheme: The encryption scheme to use, either 'ckks' or 'bfv'.
        :param scale: The global scale for CKKS scheme (default is 2^40).
        :param poly_modulus_degree: The polynomial modulus degree (default is 8192).
        :param plain_modulus: The plain modulus for BFV scheme (default is 1032193).
        :param verbose: If True, print additional information during context creation.  
        """
        self.verbose = verbose
        if scheme.lower() == "ckks":
            self.context = ts.context(
                ts.SCHEME_TYPE.CKKS,
                poly_modulus_degree=poly_modulus_degree,
                coeff_mod_bit_sizes=[40, 21, 21, 21, 21, 21, 21, 40]
            )
            self.context.global_scale = scale
            self.context.generate_galois_keys()
            self.context.generate_relin_keys()
            self.scheme = "ckks"
            if self.verbose:
                logging.info("[TenSEAL CKKS] Context created successfully.")
        elif scheme.lower() == "bfv":
            self.context = ts.context(
                ts.SCHEME_TYPE.BFV,
                poly_modulus_degree=poly_modulus_degree,
                plain_modulus=plain_modulus
            )
            self.context.generate_galois_keys()
            self.context.generate_relin_keys()
            self.scheme = "bfv"
            if self.verbose:
                logging.info("[TenSEAL BFV] Context created successfully.")
        else:
            raise ValueError("Unsupported scheme. Use 'ckks' or 'bfv'.")



    def make_context_public(self):
        """
        Make the context public.
        This method allows the context to be used for public operations,
        which is useful for scenarios where the context needs to be shared.
        """
        self.context.make_context_public()
        if self.verbose:
            logging.info("[TenSEAL] secret key removed, Context is now public.")



    def _encrypt_vector(self, vector):
        if self.scheme == "ckks":
            enc = ts.ckks_vector(self.context, vector)
            #print("[Encryption] CKKS vector encrypted.")
            return enc
        elif self.scheme == "bfv":
            vector_int = np.round(vector).astype(int)
            enc = ts.bfv_vector(self.context, vector_int)
            #print("[Encryption] BFV vector encrypted (int-converted).")
            return enc
        else:
            raise ValueError(f"Unsupported scheme: {self.scheme}")



    def encrypt(self, data):
        if isinstance(data, np.ndarray):
            if data.ndim == 1:
                #print("[Encrypt] Encrypting 1D vector...")
                return self._encrypt_vector(data)
            elif data.ndim == 2:
                #print(f"[Encrypt] Encrypting batch of {len(data)} vectors...")
                return [self._encrypt_vector(row) for row in data]
            else:
                raise ValueError("Only 1D or 2D numpy arrays are supported.")
        else:
            raise TypeError("Input must be a numpy array.")



    def can_decrypt(self,encrypted_vector):
        """
        Check if the encrypted vector can be decrypted.
        This method checks if the encrypted vector has a decrypt method,
        which indicates it is a valid encrypted object.
        :param encrypted_vector: The encrypted vector to check.
        :return: True if the vector can be decrypted, False otherwise.s
        """
        return self.context.is_private()




    def decrypt(self, encrypted_vector):
        """        Decrypt an encrypted vector.
        This method checks if the encrypted vector has a decrypt method,
        which indicates it is a valid encrypted object.
        :paramameters:
             encrypted_vector: The encrypted vector to decrypt.
        :return: The decrypted vector.
        """
        if not self.can_decrypt(encrypted_vector):
            raise ValueError("The encrypted vector cannot be decrypted with this context.")
        
        if not hasattr(encrypted_vector, "decrypt"):
            raise TypeError("Object does not support decryption.")
        decrypted = encrypted_vector.decrypt()
        #print("[Decrypt] Vector decrypted.")
        return decrypted



    def rotate(self, encrypted_vector, steps):
        """        Rotate an encrypted vector by a specified number of steps.
        This method performs a rotation operation on the encrypted vector.
        :param encrypted_vector: The encrypted vector to rotate.
        :param steps: The number of steps to rotate the vector.
        :return: The rotated encrypted vector.
        """
        if self.scheme == "ckks":
            rotated = encrypted_vector.rotate(steps)
            # print(f"[Rotate] CKKS vector rotated by {steps} steps.")
            return rotated
        elif self.scheme == "bfv":
            raise NotImplementedError("Rotation is not supported in BFV scheme.")



    def serialize(self):
        """        Serialize the context to a byte string.
        This method converts the TenSEAL context into a byte string for storage or transmission.
        :return: Serialized byte string of the context."""
        #print("[Serialize] Context serialized.")
        return self.context.serialize()

    def deserialize(self, vec_bytes):
        if self.scheme == "ckks":
            return ts.CKKSVector.load(self.context, vec_bytes)
        elif self.scheme == "bfv":
            return ts.BFVVector.load(self.context, vec_bytes)
        else:
            raise ValueError("Unsupported scheme type.")



    def save(self, path):
        """        Save the serialized context to a file.
        This method writes the serialized context to a specified file path.
        :param path: The file path where the context will be saved.
        """
        dir_path=os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path,exist_ok=True)
        with open(path, 'wb') as f:
            f.write(self.context.serialize())

        if self.verbose:
            logging.info(f"[Save] Context saved to {path}.")



    # def load(self, path, scheme = 'ckks'):
    #     """        Load the context from a file.
    #     This method reads the serialized context from a specified file path and initializes the TenSEAL context.
    #     :param path: The file path from which the context will be loaded.
    #     """
    #     with open(path, 'rb') as f:
    #         self.context = ts.context_from(f.read())

    #     # Detect and set the scheme based on context
    #     if self.context.scheme.lower() == ts.SCHEME_TYPE.CKKS:
    #         self.scheme = "ckks"
    #         self.context.global_scale = 2**40  # Needed for CKKS
    #     elif self.context.scheme_type() == ts.SCHEME_TYPE.BFV:
    #         self.scheme = "bfv"
    #     else:
    #         raise ValueError("Unsupported scheme in loaded context")
    #     if self.verbose:
    #         print(f"[Load] Context loaded from {path}.")

    def load(self, path, scheme="ckks"):
        with open(path, 'rb') as f:
            self.context = ts.context_from(f.read())

        self.scheme = scheme.lower()
        if self.scheme == "ckks":
            self.context.global_scale = 2**40  # Required to avoid 'no global scale' error
        elif self.scheme != "bfv":
            raise ValueError("Unsupported scheme specified during load().")

        if self.verbose:
            logging.info(f"[Load] Context loaded from {path} with scheme: {self.scheme.upper()}")
