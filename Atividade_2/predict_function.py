import torch


def predict(qtd_tokens: int, bigrams: dict, start_token: list, g) -> list:
    list_tokens = []
    for i in range(qtd_tokens):
        temp_dict = {}
        total = 0
        for keys, value in bigrams.items():
            if keys[0] == start_token[-1]:
                # pega o último token para assim ter a probabilidade de um token dado o token anterior
                temp_dict[(keys[0], keys[1])] = value
                total += value
        for key in temp_dict.keys():
            temp_dict[key] /= total
        probs = list(temp_dict.values())
        probs = torch.tensor(probs)
        temp_list = sorted(temp_dict.items(), key=lambda x: -x[1])
        ix = torch.multinomial(probs, num_samples=1, replacement=True, generator=g).item()
        list_tokens.extend(start_token)
        start_token = [temp_list[ix][0][1]]
    return list_tokens
