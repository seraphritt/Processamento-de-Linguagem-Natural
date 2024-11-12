import tiktoken
import unzip
import read_files
import os
import torch
import math
import time
import train_set
import generate_bigrams
import predict_function
# SOMENTE SE QUISER ESCREVER OS ARQUIVOS NO sum_corpus.txt
name_train = "sum_corpus_train.txt"
train_set.create_train_file(name_train, 8000)
with open(name_train, "r", encoding="utf-8") as file: # abre o arquivo sum_corpus.txt e lê o seu conteúdo
    whole_txt = file.read()
encoding = tiktoken.get_encoding("cl100k_base") # base do gpt-4-turbo, gpt-4, gpt-3.5-turbo
tokens = []
lista_sent = encoding.encode(whole_txt)
bigrams = generate_bigrams.generate_bigrams(lista_sent)
g = torch.Generator().manual_seed(309400321)    # definindo seed para o gerador randômico
previous_token = encoding.encode("Eu sou") # tokens iniciais para a predição do modelo bigram
ans = predict_function.predict(40, bigrams, previous_token, g)
print(encoding.decode(ans))
print(ans)
# perplexity calculation
# count = 0
# if not os.path.exists('sum_corpus_test.txt'):
#     if not os.path.exists('corpus_extracted'):
#         unzip.extract('corpus.zip', 'corpus_extracted')
#     for each in os.listdir("corpus_extracted"):
#         count += 1
#         print(count)
#         if count >= 9000:
#             data_json = read_files.read_json("corpus_extracted/" + each)
#             with open("sum_corpus_test.txt", "a", encoding="utf-8") as file:
#                 file.write(data_json["text"])
#     print(f"Arquivo de texto sum_corpus_test.txt criado")
# whole_txt = read_files.read_json("corpus_extracted/" + "240.json")["text"]
# lista_sent = encoding.encode(whole_txt)
# n = len(lista_sent)
# print(n)
# prob = 0
# start_time = time.time()
# for a, b in zip(lista_sent, lista_sent[1:]):
#     total = 0
#     temp_dict = {}
#     try:
#         for keys, values in bigrams.items():    # normalizando os valores
#             if keys[0] == a:
#                 temp_dict[(keys[0], keys[1])] = values
#                 total += values
#         for key in temp_dict.keys():
#             temp_dict[key] /= total
#         prob += math.log(temp_dict[(a, b)])
#     except KeyError:
#         prob += 1e-5 # probabilidade pequena para bigrama (novo)
# perplexity = math.exp(-prob/n)
# print(perplexity)
# end_time = time.time()
# elapsed_time = end_time - start_time
# print(f"Time elapsed: {elapsed_time:.6f} seconds")