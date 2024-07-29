import torch
from sklearn.preprocessing import LabelEncoder
from torch.nn.utils.rnn import pad_sequence
from nltk.corpus import reuters
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import nltk
from nltk.lm import Vocabulary

# import torchtext
# from torchtext.data.utils import get_tokenizer
# from torchtext.vocab import build_vocab_from_iterator


class ReutersDataset(Dataset):
    def __init__(self, texts: list[list[int]], labels: list[int]):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int):
        # This operation copies the data
        # Probably we could optimize this quite radically
        text = torch.tensor(self.texts[idx], dtype=torch.long)
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return text, label


def load_dataset():
    nltk.download("reuters")
    nltk.download("stopwords")
    nltk.download("punkt")

    # Load documents and their categories
    documents = reuters.fileids()
    categories = [reuters.categories(fileid) for fileid in documents]

    # Load document content
    data = [reuters.raw(fileid) for fileid in documents]

    text = _remove_stopwords(data)
    tokens = [sentence for sentence in text]

    vocab = Vocabulary(tokens)

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(_process_labels(categories))

    tokens = [nltk.word_tokenize(sentence) for sentence in text]
    encoded_df = pd.DataFrame(
        {
            "text": [vocab[token] for token in tokens],
            "category": label_encoder.fit_transform(labels),
        }
    )

    return encoded_df, vocab, label_encoder


def to_dataloader(
    text: pd.Series,
    labels: pd.Series,
    vocab: Vocabulary,
    shuffle: bool = True,
) -> DataLoader:
    dataset = ReutersDataset(text.tolist(), labels.tolist())
    return DataLoader(
        dataset,
        batch_size=32,
        shuffle=shuffle,
        collate_fn=lambda x: _collate_fn(x, vocab),
    )


def _collate_fn(batch: list[tuple[list[int], list[int]]], vocab: Vocabulary):
    texts, labels = zip(*batch)

    texts_padded = pad_sequence(texts, batch_first=True, padding_value=-1)
    labels = torch.tensor(labels, dtype=torch.long)

    return texts_padded, labels


def _remove_stopwords(text: list[str]) -> pd.Series:
    stop = set(nltk.corpus.stopwords.words("english"))
    return (
        pd.Series(text)
        .str.lower()
        .replace("[^\w\s]", "")
        .apply(nltk.word_tokenize)
        .apply(
            lambda sentence: " ".join([word for word in sentence if word not in stop])
        )
    )


def _process_labels(labels: list[list[str]]) -> pd.Series:
    return pd.Series(labels).apply(lambda x: x[0] if x else "unknown")
