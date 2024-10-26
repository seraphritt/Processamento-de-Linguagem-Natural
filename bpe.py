def count_pairs(li: list, d: dict) -> None:
    for i in range(len(li) - 1):
        pair = (li[i], li[i + 1])
        try:
            d[pair] += 1
        except KeyError:
            d[pair] = 1



string = "Am Anfang schuf Gott Himmel und Erde. Und die Erde war wüst und leer, und es war finster auf der Tiefe; und der Geist Gottes schwebte auf dem Wasser. Und Gott sprach: Es werde Licht! und es ward Licht. Und Gott sah, daß das Licht gut war. Da schied Gott das Licht von der Finsternis. Und Gott sah alles an, was er gemacht hatte; und siehe da, es war sehr gut. Da ward aus Abend und Morgen der sechste Tag."
print(len(string))
string = list(map(int, string.encode('utf-8')))
dicionario = {}
count_pairs(string, dicionario)
lista = []
for k, v in dicionario.items():
    lista.append((v, k))
lista.sort(reverse=True)
