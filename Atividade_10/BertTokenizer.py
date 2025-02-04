import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification


class BertTokenizer:
    def __init__(self, data_path, text_column, label_column_name, model, test_size, num_labels):
        self.data_path = data_path 
        self.text_column_name = text_column 
        self.label_column_name = label_column_name
        self.model_name = model 
        self.test_size = test_size
        self.num_labels = num_labels

    def split_datasets(self):
        df = pd.read_csv(self.data_path)
        labelencoder = preprocessing.LabelEncoder()
        labelencoder.fit(df[self.label_column_name].tolist())  
        df['label'] = labelencoder.transform(df[self.label_column_name].tolist())
        df_train, df_test = train_test_split(df, test_size=self.test_size)
        train_dataset = Dataset.from_pandas(df_train)
        test_dataset = Dataset.from_pandas(df_test)
        return train_dataset, test_dataset, df['label'], df_test

    def create_model(self):
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, 
            num_labels=self.num_labels
        )
        return model