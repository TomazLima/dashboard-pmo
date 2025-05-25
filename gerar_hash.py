import hashlib

def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

# Substitua pela sua senha desejada
senha = "minha_senha_secreta_123"
hash_gerado = make_hashes(senha)
print(f"Hash da senha '{senha}': {hash_gerado}")