import argparse
import json
from bfv.polynomial import Polynomial
from bfv.bfv import BFV, RLWE
from bfv.discrete_gauss import DiscreteGaussian
import time



def main(args):
    n = args.n
    q = args.q
    t = args.t
    sigma = 3.2  # standard deviation of the discrete Gaussian distribution

    sigma = 3.2  # standard deviation of the discrete Gaussian distribution

    # Initialize the DiscreteGaussian distribution
    distribution = DiscreteGaussian(sigma)

    # Initialize RLWE object
    rlwe = RLWE(n, q, t, distribution)

    u = rlwe.SampleFromTernaryDistribution()
    e = rlwe.SampleFromErrorDistribution()
    a = rlwe.Rq.sample_polynomial()

    bfv = BFV(rlwe)

    # Generate secret and public key
    secret_key = bfv.SecretKeyGen()
    secret_key = bfv.SecretKeyGen()

    pk_gen_start_time = time.time()
    pk_gen_start_time = time.time()

    public_key = bfv.PublicKeyGen(secret_key, e, a)

    pk_gen_end_time = time.time()
    pk_gen_end_time = time.time()
    pk_gen_elapsed_time = pk_gen_end_time - pk_gen_start_time

    print(f"Time to generate public key: {pk_gen_elapsed_time:.6f} seconds")

    # Add zeroes at the beginning of the u polynomial to make it the same degree as the public key
    u.coefficients = u.coefficients = [0] * (n - len(u.coefficients)) + u.coefficients

    # Generate message (plaintext)
    message = bfv.rlwe.Rt.sample_polynomial()

    # Add zeroes at the beginning of the message polynomial to make it the same degree as the public key
    message.coefficients = message.coefficients = [0] * (n - len(message.coefficients)) + u.coefficients


    encrypt_start_time = time.time()

    e0 = bfv.rlwe.SampleFromErrorDistribution()
    e1 = bfv.rlwe.SampleFromErrorDistribution()

    # Encrypt message
    ciphertext = bfv.PubKeyEncrypt(public_key, message, e0, e1, u)

    encrypt_end_time = time.time()
    encrypt_elapsed_time = encrypt_end_time - encrypt_start_time

    print(f"Time to encrypt the message: {encrypt_elapsed_time:.6f} seconds")

    (c0, c1) = ciphertext

    decrypt_start_time = time.time()
    # Decrypt ciphertext
    dec = bfv.Decrypt(secret_key, ciphertext)

    decrypt_end_time = time.time()
    decrypt_elapsed_time = decrypt_end_time - decrypt_start_time

    print(f"Time to decrypt the message: {decrypt_elapsed_time:.6f} seconds")
    # Ensure that message and dec match
    for i in range(len(message.coefficients)):
        assert (
            message.coefficients[i] == dec.coefficients[i]
        ), "Message and dec do not match"

    # Convert input to JSON and save to file
    with open(args.output, "w") as f:
        json.dump(
            {
                "pk0": public_key[0].into_standard_form(q).coefficients,
                "pk1": public_key[1].into_standard_form(q).coefficients,
                "m": message.into_standard_form(q).coefficients,
                "u": u.into_standard_form(q).coefficients,
                "e0": e0.into_standard_form(q).coefficients,
                "e1": e1.into_standard_form(q).coefficients,
                "c0": c0.into_standard_form(q).coefficients,
                "c1": c1.into_standard_form(q).coefficients,
                "cyclo": Polynomial(bfv.rlwe.Rq.denominator).into_standard_form(q).coefficients,
            },
            f,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate inputs for BFV zk proof circuit in json format"
    )
    parser.add_argument(
        "-n", type=int, required=True, help="Degree of f(x), must be a power of 2."
    )
    parser.add_argument(
        "-q", type=int, required=True, help="Modulus q of the ciphertext space"
    )
    parser.add_argument(
        "-t", type=int, required=True, help="Modulus t of the plaintext space."
    )
    parser.add_argument(
        "--output", type=str, default="input.json", help="Output file name"
    )

    args = parser.parse_args()
    main(args)
