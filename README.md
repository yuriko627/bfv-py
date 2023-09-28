# rlwe
rlwe implementation in Python

reference
- [Revisiting Homomorphic Encryption Schemes for Finite Fields](https://eprint.iacr.org/2012/144.pdf).
- [Somewhat Practical Fully Homomorphic Encryption](https://eprint.iacr.org/2021/204.pdf)
- [Jay's explanation](https://github.com/Janmajayamall/bfv/blob/notes/notes/BFV.md)
- [original python implemenation](https://github.com/enricobottazzi/rlwe/tree/main)


### Test

```bash
$ python3 -m unittest discover -p '*_test.py'
```

### Generate inputs for circuit

The following CLI interface is provided to generate a json file that will be used as input file for a circuit. This will be used in [zk-fhe](https://github.com/enricobottazzi/zk-fhe) for testing purposes.

The script will run through the following steps:
1. Secret key generation
2. Public key generation
3. Encryption of a random message to generate the ciphertext
4. Decryption of the ciphertext to generate the plaintext
5. Assertion of the equality between the plaintext and the original message
6. Generation of the json file

```bash
$ python3 cli.py --help
$ python3 cli.py -n 1024 -q 536870909 -t 7 --output input.json
```
