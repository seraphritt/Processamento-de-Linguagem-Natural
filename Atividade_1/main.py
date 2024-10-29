import read_files
import os
import bpe
import unzip

# SOMENTE SE QUISER ESCREVER OS ARQUIVOS NO sum_corpus.txt
unzip.extract('corpus.zip', 'corpus_extracted')
if not os.path.exists('sum_corpus.txt'):
    for each in os.listdir("corpus_extracted"):
        data_json = read_files.read_json("corpus_extracted/" + each)
        with open("sum_corpus.txt", "a", encoding="utf-8") as file:
            file.write(data_json["text"])
    print(f"Arquivo de texto sum_corpus.txt criado")


with open("sum_corpus.txt", "r", encoding="utf-8") as file: # abre o arquivo sum_corpus.txt e lê o seu conteúdo
    string = file.read()
string = list(map(int, string.encode('utf-8'))) # faz o encode do conteúdo lido para utf-8
num_merges = 20 # número de merges que será feito no algoritmo bpe
merges = bpe.merge_loop(num_merges, string) # chama a função merge_loop que faz os merges
vocab = {idx: bytes([idx]) for idx in range(256)} # inicializa o vocabulário com os 256 tokens iniciais.
for (p0, p1), idx in merges.items():
    if vocab[p0] + vocab[p1] not in vocab.values():
        vocab[idx] = vocab[p0] + vocab[p1]  # aqui os merges são representados e inseridos no vocabulário
print(vocab)  # vocabulário resultante com os merges
