import math


def calculate(text_tokenized: list, bigrams: dict) -> float:
    prob = 0
    n = len(text_tokenized)
    for a, b in zip(text_tokenized, text_tokenized[1:]):
        total = 0
        temp_dict = {}
        try:
            for keys, values in bigrams.items():  # normalizando os valores
                if keys[0] == a:
                    temp_dict[(keys[0], keys[1])] = values
                    total += values
            for key in temp_dict.keys():
                temp_dict[key] /= total
            prob += math.log(temp_dict[(a, b)])
        except KeyError:
            prob += 1e-5  # probabilidade pequena para bigrama (novo)
    perplexity = math.exp(-prob / n)
    return perplexity
