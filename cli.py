import argparse
import json
from rlwe import RLWE
from discrete_gauss import DiscreteGaussian
import time

def main(args):
    n = args.n
    q = args.q
    t = args.t
    sigma = 3 # standard deviation of the discrete Gaussian distribution
    
    # Initialize the DiscreteGaussian distribution
    distribution = DiscreteGaussian(sigma)
    
    # Initialize RLWE object
    rlwe = RLWE(n, q, t, distribution)
    
    # Generate secret and public key
    secret_key = rlwe.SecretKeyGen()

    pk_gen_start_time = time.time() 

    public_key = rlwe.PublicKeyGen(secret_key)

    pk_gen_end_time = time.time() 
    pk_gen_elapsed_time = pk_gen_end_time - pk_gen_start_time

    print(f"Time to generate public key: {pk_gen_elapsed_time:.6f} seconds")

    # Extract u polynomial from the RLWE object
    u = rlwe.u

    # Generate message (plaintext)
    message = rlwe.Rt.sample_polynomial()

    encrypt_start_time = time.time()

    # Encrypt message
    ciphertext, error = rlwe.Encrypt(public_key, message)

    encrypt_end_time = time.time()
    encrypt_elapsed_time = encrypt_end_time - encrypt_start_time

    print(f"Time to encrypt the message: {encrypt_elapsed_time:.6f} seconds")

    (c0, c1) = ciphertext
    (e0, e1) = error

    decrypt_start_time = time.time()
    # Decrypt ciphertext
    dec = rlwe.Decrypt(secret_key, ciphertext, error)

    decrypt_end_time = time.time()
    decrypt_elapsed_time = decrypt_end_time - decrypt_start_time

    print(f"Time to decrypt the message: {decrypt_elapsed_time:.6f} seconds")
    # Ensure that message and dec match
    for i in range(len(message.coefficients)):
        assert message.coefficients[i] == dec.coefficients[i], "Message and dec do not match"
    
    # Convert public key to JSON and save to file
    with open(args.output, "w") as f:
        json.dump({
            'pk0': adjust_negative_coefficients([int(coeff) for coeff in public_key[0].coefficients.tolist()], q),
            'pk1': adjust_negative_coefficients([int(coeff) for coeff in public_key[1].coefficients.tolist()], q),
            'm': adjust_negative_coefficients([int(coeff) for coeff in message.coefficients.tolist()], t),
            'u': adjust_negative_coefficients([int(coeff) for coeff in u.coefficients.tolist()], q),
            'e0': adjust_negative_coefficients([int(coeff) for coeff in e0.coefficients.tolist()], q),
            'e1': adjust_negative_coefficients([int(coeff) for coeff in e1.coefficients.tolist()], q),
            'c0': adjust_negative_coefficients([int(coeff) for coeff in c0.coefficients.tolist()], q),
            'c1': adjust_negative_coefficients([int(coeff) for coeff in c1.coefficients.tolist()], q),
        }, f)

def adjust_negative_coefficients(coefficients, modulus):
    return [(modulus + coeff if coeff < 0 else coeff) for coeff in coefficients]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate inputs for BFV zk proof circuit in json format')
    parser.add_argument('-n', type=int, required=True, help='Degree of f(x), must be a power of 2.')
    parser.add_argument('-q', type=int, required=True, help='Modulus q of the ciphertext space')
    parser.add_argument('-t', type=int, required=True, help='Modulus t of the plaintext space.')
    parser.add_argument('--output', type=str, default='input.json', help='Output file name')
    
    args = parser.parse_args()
    main(args)
