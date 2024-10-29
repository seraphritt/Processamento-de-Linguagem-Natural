def count_pairs(li: list) -> dict: # list of tokens
    d = {}
    for i in range(len(li) - 1):
        pair = (li[i], li[i + 1])
        try:
            d[pair] += 1
        except KeyError:
            d[pair] = 1
    return d    # {(x, y): quantidade}


def merge(encoded_list: list, pair: tuple, target: int) -> list:
    i = 0
    merges = []
    while i < len(encoded_list) - 1:    # se o par for o par alvo, faz o merge e anda duas posições na lista
        if (encoded_list[i], encoded_list[i + 1]) == pair:
            merges.append(target)
            i += 2
        else:
            merges.append(encoded_list[i])  # caso contrário, anda-se somente uma posição
            i += 1
    return merges


def decode(ids: list, vocab: dict) -> str:  # função de decode para testar se o bpe funciona corretamente
    tokens = b"".join(vocab[idx] for idx in ids)
    text = tokens.decode("utf-8", errors="replace")
    return text


def encode(text: str, merges: dict):    # função de encode para testar se o bpe realmente funciona corretamente
    tokens = list(text.encode("utf-8"))
    lista = []
    while len(tokens) > 1:
        d = count_pairs(tokens)
        for k, v in d.items():
            lista.append((v, k))
        lista.sort()
        pairs = lista[0][1]
        if pairs not in merges:
            break
        idx = merges[pairs]
        tokens = merge(tokens, pairs, idx)
    return tokens


def merge_loop(qtd_merges: int, encoded_string: list) -> dict:  # função de loop que executa o merge (qtd_merges) vezes
    merges = {}
    for i in range(qtd_merges):
        li = []
        stats = count_pairs(encoded_string)
        for k, v in stats.items():
            li.append((v, k))
        li.sort(reverse=True)
        pair = li[0][1]
        target = 256 + i
        print(f'merging {pair} into a new token {target}')
        encoded_string = merge(encoded_string, pair, target) # after merge
        merges[pair] = target
    return merges