import os

import numpy as np
import requests
import tensorflow as tf
import tensorflow.keras as tfk
from kmeseg.utils.config import LOOK_BACK, word_segmentation_model_url


class Segmentation:
    def __init__(self):
        word_segment_url = word_segmentation_model_url
        response = requests.get(word_segment_url, allow_redirects=True)
        with open("my_model.hdf5", "wb") as f_hdf5, requests.Session() as req:
            f_hdf5.write(response.content)

            resp = req.get(word_segmentation_model_url)
            self.CHAR_INDICES = resp.json()

        self.model = tfk.models.load_model("my_model.hdf5")
        os.remove("my_model.hdf5")

        self.LOOK_BACK = LOOK_BACK

    def preprocessing_text(self, raw_text):
        """
        take unseen (testing) text and encode it with CHAR_DICT
        //It's like create_dataset() but not return label
        return preprocessed text
        """
        X = []
        data = [self.CHAR_INDICES["<pad>"]] * self.LOOK_BACK
        for char in raw_text:
            char = (
                char if char in self.CHAR_INDICES else "<unk>"
            )  # check char in dictionary
            data = data[1:] + [self.CHAR_INDICES[char]]  # X data
            X.append(data)
        return np.array(X)

    def predict(self, preprocessed_text):
        pred = self.model.predict(preprocessed_text)
        class_ = tf.argmax(pred, axis=-1).numpy()

        return class_

    def word_segmentation(self, text):
        preprocessed_text = self.preprocessing_text(text)
        class_ = self.predict(preprocessed_text)
        class_[0] = 1
        class_ = np.append(class_, 1)

        cut_indexes = [i for i, value in enumerate(class_) if value == 1]
        words = [
            text[cut_indexes[i] : cut_indexes[i + 1]]
            for i in range(len(cut_indexes) - 1)
        ]

        join_word = "|".join(words)

        return words, join_word


class Tokenizer:
    def __init__(self):
        self.word2index = {"<pad>": 0, "<start>": 1, "<end>": 2, "<unk>": 3}
        self.word2count = {"<pad>": 0, "<start>": 0, "<end>": 0, "<unk>": 0}
        self.index2word = {0: "<pad>", 1: "<start>", 2: "<end>", 3: "<unk>"}
        self.num_words = 4
        self.num_sentences = 0
        self.longest_sentences = 0
        self.kme_segment = Segmentation()

    def add_word(self, word):
        if word not in self.word2index:

            # First entry of word into vocabulary
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:

            # Word exists; increase word count
            self.word2count[word] += 1

    def fit_on_text(self, sentences):
        for sentence in sentences:

            # Model predict
            segmented_text, _ = self.kme_segment.word_segmentation(sentence)

            # Add <start> at start and <end> at end of each sentences
            segmented_text = np.concatenate(
                (["<start>"], segmented_text, ["<end>"])
            )

            for word in segmented_text:
                self.add_word(word)

    def text_to_sequences(self, sentences, method_pad=None):
        sequences_arr = []
        for sentence in sentences:

            # Model predict
            segmented_text, _ = self.kme_segment.word_segmentation(sentence)
            tokenize_text_arr = []

            for word in segmented_text:
                try:
                    tokenize_text_arr.append(self.word2index[word])
                except KeyError:

                    # use <unk> key for word that never met
                    tokenize_text_arr.append(self.word2index["<unk>"])

            sequences_arr.append(tokenize_text_arr)
        if method_pad:
            # Make zero padding by Maximum length of sentence
            sequences_padded = tfk.preprocessing.sequence.pad_sequences(
                sequences_arr, padding=method_pad
            )
            return sequences_padded
        return sequences_arr

    def sequences_to_text(self, sequences):
        texts_arr = []
        for sequence in sequences:
            index_arr = []
            for index in sequence:
                index_arr.append(self.index2word[index])
            texts_arr.append(index_arr)
        return texts_arr
