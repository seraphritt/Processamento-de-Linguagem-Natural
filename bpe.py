def count_pairs(li: list, d: dict) -> None:
    for i in range(len(li) - 1):
        pair = (li[i], li[i + 1])
        try:
            d[pair] += 1
        except KeyError:
            d[pair] = 1


def merge(merges: dict, encoded_list: list, pair: tuple, target: int) -> dict:
    i = 0
    while i < len(encoded_list) - 1:
        if (encoded_list[i], encoded_list[i + 1]) == pair:
            merges[pair] = target
            # encoded_list[i] = target
            # encoded_list.remove(encoded_list[i + 1])
        i += 1
    return merges


string = "Am Anfang schuf Gott Himmel und Erde. Und die Erde war wüst und leer, und es war finster auf der Tiefe; und der Geist Gottes schwebte auf dem Wasser. Und Gott sprach: Es werde Licht! und es ward Licht. Und Gott sah, daß das Licht gut war. Da schied Gott das Licht von der Finsternis. Und Gott sah alles an, was er gemacht hatte; und siehe da, es war sehr gut. Da ward aus Abend und Morgen der sechste Tag."
print(len(string)) # tokens
string = list(map(int, string.encode('utf-8')))
print(string)
dicionario = {}
count_pairs(string, dicionario)
lista = []
for k, v in dicionario.items():
    lista.append((v, k))
lista.sort(reverse=True)
print(len(string))
m = {}
merges = merge(m, string, (100, 32), 512) # after merge
# print(f'Merging (, ) into a new token {1}')
vocab = {idx: bytes([idx]) for idx in range(256)}
for (p0, p1), idx in merges.items():
    vocab[idx] = vocab[p0] + vocab[p1]


def decode(ids: list) -> str:
    tokens = b"".join(vocab[idx] for idx in ids)
    text = tokens.decode("utf-8", errors="replace")
    return text


print(decode([255]))


def encode(text: str):
    tokens = list(text.encode("utf-8"))
    d = {}
    lista = []
    while len(tokens) > 1:
        count_pairs(tokens, d)
        for k, v in d.items():
            lista.append((v, k))
        print(lista)
        lista.sort()
        pairs = lista[0][1]
        if pairs not in merges:
            break
        idx = merges[pairs]
        tokens = merge({}, tokens, pairs, idx)
    return tokens


print(encode("hello world!"))
print(decode(encode("hello worldalsçcçlsa!")))
