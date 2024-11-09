import tiktoken
import unzip
import read_files
import os
import torch
# SOMENTE SE QUISER ESCREVER OS ARQUIVOS NO sum_corpus.txt
if not os.path.exists('sum_corpus.txt'):
    unzip.extract('corpus.zip', 'corpus_extracted')
    for each in os.listdir("corpus_extracted"):
        data_json = read_files.read_json("corpus_extracted/" + each)
        with open("sum_corpus.txt", "a", encoding="utf-8") as file:
            file.write(data_json["text"])
    print(f"Arquivo de texto sum_corpus.txt criado")
with open("sum_corpus.txt", "r", encoding="utf-8") as file: # abre o arquivo sum_corpus.txt e lê o seu conteúdo
    whole_txt = file.read()
encoding = tiktoken.get_encoding("cl100k_base")
tokens = []
lista_sent = encoding.encode(whole_txt)
# print(lista_sent)
bigrams = {}
for a, b in zip(lista_sent, lista_sent[1:]):
    try:
        bigrams[(a, b)] += 1
    except KeyError:
        bigrams.update({(a, b): 1})
g = torch.Generator().manual_seed(309400321)
previous_token = encoding.encode("Amigo")
ans = []
for i in range(40):
    temp_dict = {}
    total = 0
    for keys, value in bigrams.items():
        if keys[0] == previous_token[-1]:
            # pega o último token para assim ter a probabilidade de um token dado o token anterior
            temp_dict[(keys[0], keys[1])] = value
            total += value
    for key in temp_dict.keys():
        temp_dict[key] /= total
    probs = list(temp_dict.values())
    probs = torch.tensor(probs)
    temp_list = sorted(temp_dict.items(), key=lambda x: -x[1])
    ix = torch.multinomial(probs, num_samples=1, replacement=True, generator=g).item()
    ans.extend(previous_token)
    previous_token = [temp_list[ix][0][1]]
print(encoding.decode(ans))
print(ans)