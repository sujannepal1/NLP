import math
import os
from collections import Counter

import numpy as np
import pandas as pd


class Document:
    def __init__(self, text, name=""):
        self.name = name
        self.text = text
        self.words: list = text.split()
        self.formatted_words = []

        for word in self.words:
            word = word.lower()  # Convert to lowercase for case-insensitive counting
            word = word.strip()  # Remove leading/trailing whitespace
            word = word.strip(".,!?;:\"'()[]{}")  # Remove punctuation
            if word:
                self.formatted_words.append(word)

        self.vocabulary: set = (
            set(self.formatted_words) if self.formatted_words else set()
        )
        self.formatted_word_length = len(self.formatted_words)

    def term_frequency(self):
        total = len(self.formatted_words)
        if total == 0:
            return {}
        counts = Counter(self.formatted_words)

        return {word: count / total for word, count in counts.items()}

    def word_frequency(self):
        return Counter(self.formatted_words)

    def co_occurrence_matrix(self, window_size=1):
        length_of_unique_words = len(self.vocabulary)
        co_matrix = np.zeros(
            (length_of_unique_words, length_of_unique_words), dtype=int
        )
        sorted_vocabulary = sorted(self.vocabulary)
        label_of_vocabulary = {
            word: index for index, word in enumerate(sorted_vocabulary)
        }
        print("label of vocab : ", label_of_vocabulary)
        # need to populate the matrix with co-occurrence counts
        token_indices = dict(enumerate(self.words))
        print("token indices :", token_indices)
        for index, value in token_indices.items():
            try:
                index_of_current_word = label_of_vocabulary[value]
            except KeyError:
                value = value.strip()
                value = value.lower()
                value = value.strip()
                value = value.strip(".,!?;:\"'()[]{}")
                index_of_current_word = label_of_vocabulary[value.lower()]
            except Exception:  # noqa: BLE001
                print(f"Error: missing key in the vocabulary for the word {value}")
                continue

            starting_index = max(index - window_size, 0)
            stopping_index = min(index + window_size, length_of_unique_words - 1)
            for i in range(starting_index, stopping_index):
                if i != index:
                    token = token_indices[i].lower()
                    token = token.strip()
                    token = token.strip(".,!?;:\"'()[]{}")
                    index_of_word_in_vocab = list(label_of_vocabulary.keys()).index(
                        token
                    )
                    co_matrix[index_of_current_word, index_of_word_in_vocab] += 1
                    co_matrix[index_of_word_in_vocab, index_of_current_word] += 1

        df = pd.DataFrame(
            co_matrix, index=sorted_vocabulary, columns=sorted_vocabulary, dtype=int
        )
        print(df)

        # Source - https://stackoverflow.com/a/54357705
        # Posted by Julien
        # Retrieved 2026-08-17, License - CC BY-SA 4.0
        zero_values_count = (df.values == 0).sum()
        print("value of zero counts in the sparse matrix", zero_values_count)
        print(
            f"Out of all the columns and rows we have {length_of_unique_words**2 - zero_values_count} values populated"
        )
        # uncomment later line to find more about each column max value
        # print("max value of each column", df.max())
        return co_matrix


class Corpus:
    def __init__(self, data_path):
        self.data_path = data_path
        self.documents: list[Document] = []
        self.load_directory()
        self.document_count = len(self.documents)

    def add_document(self, document: Document):
        self.documents.append(document)

    def load_directory(self):
        for filename in os.listdir(self.data_path):
            if filename.endswith(".txt"):
                print(f"Loading document: {filename}")
                file_path = os.path.join(self.data_path, filename)

                with open(file_path, "r", encoding="utf-8") as file:
                    text = file.read()

                self.add_document(Document(text, filename))
                break
            # other type of documents like pdf, csv, etc. can be added here

    def vocabulary(self):
        vocab = set()
        for document in self.documents:
            vocab.update(document.vocabulary)
        return vocab

    def total_words(self):
        return sum(document.formatted_word_length for document in self.documents)

    def word_frequencies(self):
        frequencies = Counter()

        for document in self.documents:
            frequencies.update(document.word_frequency())

        return frequencies

    def inverse_document_frequency(self, word, total_docs):
        doc_count = sum(1 for document in self.documents if word in document.vocabulary)
        if doc_count == 0:
            return 0
        return math.log(total_docs / doc_count)

    def tf_idf(self):
        total_docs = self.document_count
        tf_idf_scores: dict = {}
        for document in self.documents:
            tf_idf_scores[document.name] = {}
            tf_scores = document.term_frequency()
            for word, tf in tf_scores.items():
                idf = self.inverse_document_frequency(word, total_docs)
                tf_idf_scores[document.name][word] = tf * idf

        return tf_idf_scores


class NGram:
    def __init__(self, content: Corpus | Document, n=1) -> None:
        self.content = content
        self.n = n
        self.n_gram_list = []

    def decouple(self):
        if isinstance(self.content, Corpus):
            for doc in self.content.documents:
                self.n_gram_value(self.n_gram_list, self.n, doc)
        elif isinstance(self.content, Document):
            self.n_gram_value(self.n_gram_list, self.n, self.content)
        else:
            raise NotImplementedError(f"{type(self.content)} not implemented")
        return self.n_gram_list

    def n_gram_value(self, n_gram_list, n_value, document):
        words = document.words

        n_gram_list.extend(
            tuple(words[i : i + n_value]) for i in range(len(words) - n_value + 1)
        )


DATA_PATH = "data/nepali_news"

corpus = Corpus(DATA_PATH)

corpus.documents[0].co_occurrence_matrix()
# Coccurrence_matrix = CoccurrenceMatrix(corpus.vocabulary())

print(corpus.documents[0].words)

print(NGram(corpus, 2).decouple())
