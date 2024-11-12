def generate_bigrams(lista_tokenized) -> dict:
    bigrams = {}
    for a, b in zip(lista_tokenized, lista_tokenized[1:]):
        try:
            bigrams[(a, b)] += 1
        except KeyError:
            bigrams.update({(a, b): 1})
    return bigrams