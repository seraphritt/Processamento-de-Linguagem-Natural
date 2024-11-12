import os
import read_files
import unzip


def create_train_file(name_train_file: str, qtd_train: int) -> None:
    # a qtd_train é definida e depois disso o resto vai ser qtd_test, ou seja, será parte do conjunto de teste
    count = 0
    if not os.path.exists(name_train_file):
        if not os.path.exists('corpus_extracted'):
            unzip.extract('corpus.zip', 'corpus_extracted')
        for each in os.listdir("corpus_extracted"):
            if count == qtd_train:
                break
            count += 1
            data_json = read_files.read_json("corpus_extracted/" + each)
            with open(name_train_file, "a", encoding="utf-8") as file:
                file.write(data_json["text"])
        print(f"Arquivo de texto {name_train_file} criado")



