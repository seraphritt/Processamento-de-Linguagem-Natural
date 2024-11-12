import tiktoken
import read_files
import os
import torch
import train_set
import generate_bigrams
import predict_function
import random
import perplexity

# generating train set
name_train = "sum_corpus_train.txt"
train_set.create_train_file(name_train, 8000)
with open(name_train, "r", encoding="utf-8") as file: # abre o arquivo sum_corpus.txt e lê o seu conteúdo
    whole_txt = file.read()
encoding = tiktoken.get_encoding("cl100k_base") # base do gpt-4-turbo, gpt-4, gpt-3.5-turbo
lista_sent = encoding.encode(whole_txt)
bigrams = generate_bigrams.generate_bigrams(lista_sent)
g = torch.Generator().manual_seed(309400321)    # definindo seed para o gerador randômico
previous_token = encoding.encode("Bom dia") # tokens iniciais para a predição do modelo bigram

# predicting 40 tokens
ans = predict_function.predict(40, bigrams, previous_token, g)
print(encoding.decode(ans))

# perplexity calculation
lista_path = os.listdir("corpus_extracted")
random_int = random.randint(8000, len(lista_path))
# pega um arquivo entre os 2000 arquivos restantes pertencentes ao conjunto de teste
whole_txt = read_files.read_json("corpus_extracted/" + lista_path[random_int])["text"]
print(whole_txt)
lista_sent = encoding.encode(whole_txt)
print(f"Perplexidade: {perplexity.calculate(lista_sent, bigrams)}")
