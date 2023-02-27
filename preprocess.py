import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

class Preprocess():
    """Data preprocessing."""

    def preprocess(self, data, label2id, sequence_index_dict, max_seq):
        """Preprocess input and output data for the model.

        Args:
            data: Dataset.
            label2id: Mapping dictionary of labels and IDs.
            sequence_index_dict: Dictionary of items and LCU IDs.
            max_seq: Maximum sequence length.

        Returns:
            x: Tuple of input data.
            y: One-hot vector of output data.
        """

        user, item, item_user_sequence = self.get_data(data)
        item_user_sequence_index = self.insert_sequence_index(item, sequence_index_dict)
        item_user_sequence_index = self.padding(item_user_sequence_index, max_seq)
        item_user_sequence = self.padding(item_user_sequence, max_seq)
        x = (user, item, item_user_sequence, item_user_sequence_index)
        y = self.one_hot(data, label2id)
        return x, y


    def get_data(self, data):
        """Obtain all data elements.

        Args:
            data: Data object.

        Returns:
            user: Users in the dataset.
            item: Items in the dataset.
            item_user_sequence: User sequences of items in the dataset.
        """

        user = data['userId'].values
        item = data['itemId'].values
        item_user_sequence = data['item_user_sequence'].values
        return user, item, item_user_sequence


    def generate_sequence_index(self, item, sequence, max_seq):
        """Generate new compact ID for historical sequences (LCU IDs).

        For example:
            Titanic: [Sara,Joy,Amy] --> 1: ['1',2,3]
            Thor: [Sara,Amy] --> 2: ['4',5]

            '1' represents an ID for Titanic-Sara LCU and Sara-Titanic LCU.
            '4' represents an ID for Thor-Sara LCU and Sara-Thor LCU.

        Args:
            item: Item.
            sequence: Item historical sequence.
            max_seq: Maximum length of historical sequence.

        Returns:
            sequence_index_dict: Dictionary of items and LCU IDs.
            count: Total number of LCUs.
        """

        start = 1
        sequence_index_dict = {}
        count = 0
        for i in range(len(item)):
            if len(sequence[i]) > max_seq:
                sequence[i] = sequence[i][:max_seq]
            count += len(sequence[i])
            stop = start + len(sequence[i])
            seq_index = np.array(range(start, stop))
            sequence_index_dict[item[i]] = seq_index
            start = stop
        return sequence_index_dict, count


    def insert_sequence_index(self, item, sequence_index_dict):
        """Store LCU IDs into array.

        Args:
            item: Items.
            sequence_index_dict: Dictionary of item and LCU IDs.

        Returns:
            Array of LCU IDs.
        """

        sequence_index = []
        for i in item:
            sequence_index.append(sequence_index_dict[i])
        return np.array(sequence_index)


    def padding(self, sequence, max_seq):
        """Padding sequences.

        Args:
            sequence: List of sequences.
            max_seq: Maximum length of all sequences.

        Return:
            sequence: Sequences with length `max_seq`.
        """

        sequence = pad_sequences(sequence, max_seq, padding='post', truncating='post')
        return sequence


    def generate_label_dict(self, label_list):
        """Create mapping dictionary between labels and IDs.

        Args:
            label_list: List of labels.

        Returns:
            label2id: Mapping dictionary of labels and IDs.
            id2label: Mapping dictionary of IDs and labels.
        """

        label2id = {}
        id2label = {}
        for i,label in enumerate(label_list):
            label2id[label] = i
            id2label[i] = label
        return label2id, id2label


    def one_hot(self, data, label2id):
        """Create One-Hot vector for all labels in the dataset.

        Args:
            data: Data.
            label2id: Mapping dictionary of labels and IDs.

        Returns:
            onehot: One-Hot vector.
        """

        onehot = data['rating'].apply(lambda x: label2id[x]).tolist()
        onehot = to_categorical(onehot, num_classes=len(label2id))
        return onehot
