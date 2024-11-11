import tiktoken
import unzip
import read_files
import os
import torch
import math
import time

# SOMENTE SE QUISER ESCREVER OS ARQUIVOS NO sum_corpus.txt
count = 0
if not os.path.exists('sum_corpus_train.txt'):
    if not os.path.exists('corpus_extracted'):
        unzip.extract('corpus.zip', 'corpus_extracted')
    for each in os.listdir("corpus_extracted"):
        if count == 9000:
            break
        count += 1
        data_json = read_files.read_json("corpus_extracted/" + each)
        with open("sum_corpus_train.txt", "a", encoding="utf-8") as file:
            file.write(data_json["text"])
    print(f"Arquivo de texto sum_corpus.txt criado")
with open("sum_corpus_train.txt", "r", encoding="utf-8") as file: # abre o arquivo sum_corpus.txt e lê o seu conteúdo
    whole_txt = file.read()
encoding = tiktoken.get_encoding("cl100k_base") # base do gpt-4-turbo, gpt-4, gpt-3.5-turbo
tokens = []
lista_sent = encoding.encode(whole_txt)
# print(lista_sent)
bigrams = {}
dict_keys = {}
for a, b in zip(lista_sent, lista_sent[1:]):
    try:
        bigrams[(a, b)] += 1
    except KeyError:
        bigrams.update({(a, b): 1})
    try:
        dict_keys[a].add((a, b))
    except KeyError:
        dict_keys[a] = {(a, b)}
print(dict_keys)
g = torch.Generator().manual_seed(309400321)
previous_token = encoding.encode("Bom dia")
print(previous_token)
print(previous_token[-1])
ans = []
# predict funcion
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
# perplexity calculation
count = 0
if not os.path.exists('sum_corpus_test.txt'):
    if not os.path.exists('corpus_extracted'):
        unzip.extract('corpus.zip', 'corpus_extracted')
    for each in os.listdir("corpus_extracted"):
        count += 1
        print(count)
        if count >= 9000:
            data_json = read_files.read_json("corpus_extracted/" + each)
            with open("sum_corpus_test.txt", "a", encoding="utf-8") as file:
                file.write(data_json["text"])
    print(f"Arquivo de texto sum_corpus_test.txt criado")
whole_txt = read_files.read_json("corpus_extracted/" + "240.json")["text"]
lista_sent = encoding.encode(whole_txt)
n = len(lista_sent)
print(n)
prob = 0
start_time = time.time()
for a, b in zip(lista_sent, lista_sent[1:]):
    total = 0
    temp_dict = {}
    try:
        for keys, values in bigrams.items():    # normalizando os valores
            if keys[0] == a:
                temp_dict[(keys[0], keys[1])] = values
                total += values
        for key in temp_dict.keys():
            temp_dict[key] /= total
        prob += math.log(temp_dict[(a, b)])
    except KeyError:
        prob += 1e-5 # probabilidade pequena para bigrama (novo)
perplexity = math.exp(-prob/n)
print(perplexity)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Time elapsed: {elapsed_time:.6f} seconds")