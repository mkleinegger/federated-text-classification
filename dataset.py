import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from sklearn.preprocessing import LabelEncoder
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import Vocab
from nltk.corpus import reuters
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import nltk


class ReutersDataset(Dataset):
    def __init__(self, texts: list[list[int]], labels: list[int]):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int):
        # This operation copies the data and moves it to the GPU
        # Probably we could optimize this quite radically
        print("Copying data")
        text = torch.tensor(self.texts[idx], dtype=torch.long)
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return text, label


def load_dataset():
    nltk.download("reuters")
    nltk.download("stopwords")

    # Load documents and their categories
    documents = reuters.fileids()
    categories = [reuters.categories(fileid) for fileid in documents]

    # Load document content
    data = [reuters.raw(fileid) for fileid in documents]

    # Create a DataFrame
    df = pd.DataFrame(
        {
            "text": _remove_stopwords(data),
            "category": _process_labels(categories),
        }
    )

    # Tokenize and build vocabulary
    tokenizer = get_tokenizer("basic_english")
    tokens = [tokenizer(text) for text in df["text"]]

    vocab = build_vocab_from_iterator(tokens, specials=["<unk>", "<pad>"])
    vocab.set_default_index(vocab["<unk>"])

    label_encoder = LabelEncoder()
    encoded_df = pd.DataFrame(
        {
            "text": [vocab(token) for token in tokens],
            "category": label_encoder.fit_transform(df["category"]),
        }
    )

    return encoded_df, vocab, label_encoder


def to_dataloader(text: pd.Series, labels: pd.Series, vocab: Vocab, shuffle: bool = True) -> DataLoader:
    dataset = ReutersDataset(text.tolist(), labels.tolist())
    return DataLoader(
        dataset,
        batch_size=32,
        shuffle=shuffle,
        collate_fn=lambda x: _collate_fn(x, vocab),
        generator=torch.Generator(device=torch.device("mps"))
    )


def _collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor]], vocab: Vocab):
    texts, labels = zip(*batch)
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=vocab["<pad>"])
    print("Copying data")
    return texts_padded, torch.tensor(labels)


def _remove_stopwords(text: list[str]) -> pd.Series:
    stop = set(nltk.corpus.stopwords.words("english"))
    return (
        pd.Series(text)
        .str.lower()
        .replace("[^\w\s]", "")
        .apply(
            lambda x: " ".join([word for word in x.split() if word not in stop]),
        )
    )


def _process_labels(labels: list[list[str]]) -> pd.Series:
    return pd.Series(labels).apply(lambda x: x[0] if x else "unknown")
