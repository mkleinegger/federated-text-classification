import nltk
import torch
import pandas as pd
from nltk.lm import Vocabulary
from nltk.corpus import reuters
from sklearn.preprocessing import LabelEncoder
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, random_split


class ReutersDataset(Dataset):
    def __init__(self, texts: list[list[int]], labels: list[int], maxlen: int):
        self.texts = [text[:maxlen] for text in texts]
        self.labels = labels
        self.maxlen = maxlen

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int):
        text = torch.tensor(self.texts[idx], dtype=torch.long)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return text, label


def __collate_fn(batch: list[tuple[list[int], list[int]]], vocab: Vocabulary):
    texts, labels = zip(*batch)
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=vocab["<pad>"])
    labels = torch.tensor(labels, dtype=torch.long)
    return texts_padded, labels


def __remove_stopwords(text: list[str]) -> pd.Series:
    stop = set(nltk.corpus.stopwords.words("english"))
    return (
        pd.Series(text)
        .str.lower()
        .replace(r"[^\w\s]", "", regex=True)
        .apply(nltk.word_tokenize)
        .apply(
            lambda sentence: " ".join([word for word in sentence if word not in stop])
        )
    )


def __process_labels(labels: list[list[str]]) -> pd.Series:
    return pd.Series(labels).apply(lambda x: x[0] if x else "unknown")


def load_dataset(maxlen: int):
    nltk.download("reuters")
    nltk.download("stopwords")
    # nltk.download("punkt")

    # Load documents and their categories
    documents = reuters.fileids()
    categories = [reuters.categories(fileid) for fileid in documents]

    # Load document content
    data = [reuters.raw(fileid) for fileid in documents]

    text = __remove_stopwords(data)
    tokens = [nltk.word_tokenize(sentence) for sentence in text]

    # Flatten the list of tokens for vocabulary creation
    flat_tokens = [token for sentence in tokens for token in sentence]
    vocab = Vocabulary(flat_tokens)

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(__process_labels(categories))

    # Add a padding token to the vocabulary
    vocab.update(["<pad>"])
    encoded_texts = [[vocab[token] for token in sentence] for sentence in tokens]

    # Apply maxlen to encoded texts
    encoded_texts = [text[:maxlen] for text in encoded_texts]

    encoded_df = pd.DataFrame(
        {
            "text": encoded_texts,
            "category": labels,
        }
    )

    return encoded_df, vocab, label_encoder


def partition_dataset(
    train_loader,
    test_loader,
    vocab: Vocabulary,
    num_partitions: int,
    batch_size: int,
    val_ratio: float = 0.1,
):
    """This function partitions the training set into N disjoint
    subsets, each will become the local dataset of a client. This
    function also subsequently partitions each training set partition
    into train and validation. The test set is left intact and will
    be used by the central server to assess the performance of the
    global model."""

    # Extract dataset from DataLoader
    train_dataset = train_loader.dataset
    test_dataset = test_loader.dataset

    # Calculate partition lengths
    num_images = len(train_dataset)
    partition_len = [num_images // num_partitions] * num_partitions
    partition_len[-1] += (
        num_images % num_partitions
    )  # Add remainder to the last partition

    # Split the dataset into `num_partitions`
    trainsets = random_split(
        train_dataset, partition_len, torch.Generator().manual_seed(2023)
    )

    # Create DataLoaders with train+val support
    trainloaders = []
    valloaders = []
    for trainset in trainsets:
        num_total = len(trainset)
        num_val = int(val_ratio * num_total)
        num_train = num_total - num_val

        train_subset, val_subset = random_split(
            trainset, [num_train, num_val], torch.Generator().manual_seed(2023)
        )

        trainloaders.append(
            DataLoader(
                train_subset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                collate_fn=lambda x: __collate_fn(x, vocab),
            )
        )
        valloaders.append(
            DataLoader(
                val_subset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=lambda x: __collate_fn(x, vocab),
            )
        )

    # Create DataLoader for the test set
    testloader = DataLoader(
        test_dataset, batch_size=128, collate_fn=lambda x: __collate_fn(x, vocab)
    )

    return trainloaders, valloaders, testloader


def create_dataloader(
    text: pd.Series,
    labels: pd.Series,
    vocab: Vocabulary,
    maxlen: int,
    shuffle: bool = True,
) -> DataLoader:
    dataset = ReutersDataset(text.tolist(), labels.tolist(), maxlen)
    return DataLoader(
        dataset,
        batch_size=32,
        shuffle=shuffle,
        collate_fn=lambda x: __collate_fn(x, vocab),
    )
