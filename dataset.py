from sklearn.model_selection import train_test_split
from datetime import datetime
from time import time
from json import loads
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
import argparse


def parse_args():
    """Arguments for building the dataset via command-line interfaces."""

    parser = argparse.ArgumentParser(description="Build Dataset")

    parser.add_argument('--data_path', nargs='?', default='Data/',
                        help='Data path')
    parser.add_argument('--dataset', nargs='?', default='ml-1m',
                        help='Dataset name')
    parser.add_argument('--min_seq', type=int, default=20,
                        help='Minimun sequence length')
    parser.add_argument('--test_ratio', type=float, default=0.2,
                        help='Test ratio for spliting train and test')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--data_percent', type=float, default=1.0,
                        help='Percentage of data used to build a dataset')
    parser.add_argument('--show_details', nargs='?', default='',
                        help='Show data details by inputting the data version')

    return parser.parse_args()


def get_time():
    """Get the current time."""
    return '[{}]:>\t'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


class Dataset():
    """Building dataset."""

    def __init__(self, data_path):
        """Constructor for `Dataset`.

        Args:
            data_path: Path of data.
        """

        self.data_path = data_path
        self.data = pd.DataFrame()
        self.df_train = pd.DataFrame()
        self.df_test = pd.DataFrame()
        self.ui_seq_df = pd.DataFrame()
        self.iu_seq_df = pd.DataFrame()
        self.users = set()
        self.items = set()
        self.labels = set()


    def create_sequence(self, seq_type):
        """Create historical sequences of users/items.

        Args:
            seq_type: Type of historical sequence:
                `ui` is user historical sequence.
                `iu` is item historical sequence.

        Returns:
            seq_df: Dataframe contained historical sequences.
        """

        def find_sequence(key, value, seq_col):
            """Gather all historical sequences of users/items.

            Args:
                key: User or item.
                value: Historical interactions and timestamp.
                seq_col: Column name of historical interactions.

            Returns:
                unique_keys: User or item.
                result: Historical sequences
            """

            unique_keys, keys_as_int = np.unique(key, return_inverse=True)
            print(get_time(), '\t\t| creating list of indice ...')
            indices = [[] for _ in range(len(unique_keys))]
            log = 50000
            for i, k in tqdm(enumerate(keys_as_int)):
                if i % log == 0:
                    temp_log = get_time() + '\t\t| generating indice at record {}'.format(i)
                    tqdm.write(temp_log)
                indices[k].append(i)
            result = [[]]*len(indices)
            for k, idx in tqdm(enumerate(indices)):
                if k % log == 0:
                    temp_log = get_time() + '\t\t| grouping sequence at item {}'.format(k)
                    tqdm.write(temp_log)
                v = np.sort(value[idx], order='timestamp')[seq_col]
                result[k] = v
            return unique_keys, result

         # Create user historical sequence.
        if seq_type == 'ui':
            key = self.data['userId'].values
            value = np.array(self.data[['itemId', 'timestamp']].to_records())[['itemId', 'timestamp']]
            key, value = find_sequence(key, value, 'itemId')
            seq_df = pd.DataFrame()
            seq_df['userId'] = key
            seq_df['user_item_sequence'] = value

        # Create item historical sequence.
        if seq_type == 'iu':
            key = self.data['itemId'].values
            value = np.array(self.data[['userId', 'timestamp']].to_records())[['userId', 'timestamp']]
            key, value = find_sequence(key, value, 'userId')
            seq_df = pd.DataFrame()
            seq_df['itemId'] = key
            seq_df['item_user_sequence'] = value

        return seq_df


    def find_unfrequent(self, mode, min_seq):
        """Find unfrequent users/items.

        Args:
            mode: Whether to find unfrequent users or unfrequent items.
            min_seq: Frequency threshold.

        Returns:
            unfreq_item: Unfrequent users/items with fewer interactions than the threshold.
        """

        if mode == 'user':
            # Find sequence lenght of user.
            self.ui_seq_df['len_user_item_sequence'] = self.ui_seq_df['user_item_sequence'].apply(len)
            # Find unfrequent user.
            unfreq_user = self.ui_seq_df[self.ui_seq_df['len_user_item_sequence'] < min_seq]['userId'].tolist()
            return unfreq_user

        if mode == 'item':
            # Find sequence lenght of item.
            self.iu_seq_df['len_item_user_sequence'] = self.iu_seq_df['item_user_sequence'].apply(len)
            # Find unfrequent item.
            unfreq_item = self.iu_seq_df[self.iu_seq_df['len_item_user_sequence'] < min_seq]['itemId'].tolist()
            return unfreq_item


    def remove_unfrequent(self, data, unfreq_list, mode):
        """Remove unfrequent users/items.

        Args:
            data: Data.
            unfreq_list: List of unfrequent users/items that will be removed.
            mode: Whether to remove unfrequent users or unfrequent items.

        Returns:
            data: Data that already removed unfrequent users/items."""

        # Remove unfrequent user and item
        if mode == 'user':
            data = data[~data['userId'].isin(unfreq_list)]

        if mode == 'item':
            data = data[~data['itemId'].isin(unfreq_list)]

        return data


    def insert_sequence(self):
        """Insert historical sequences into dataframe."""

        item_user_sequence_dict = self.iu_seq_df.set_index('itemId').T.to_dict('list')
        item_user_sequence = []
        items = self.data['itemId'].values
        log = 50000
        for i in tqdm(range(len(items))):
            if i % log == 0:
                temp_log = get_time() + '\t\t| inserting sequence at record {}'.format(i)
                tqdm.write(temp_log)
            item_user_sequence.append(item_user_sequence_dict[items[i]][0])
        self.data['item_user_sequence'] = item_user_sequence


    def create_vocab(self, data, mode):
        """List all unique users and items."""

        if mode == 'user':
            list(map(lambda users: self.users.update(users), data['item_user_sequence'].tolist()))

        if mode == 'item':
            self.items.update(data['itemId'].unique())


    def generate_dicts(self, data_arr):
        """Generate the dictionaries.

        Args:
            data_arr: List of item and ID pairs.

        Returns:
            item2id: Mapping dictionary of items and IDs.
            id2item: Mapping dictionary of IDs and item.
        """

        item2id = {}
        id2item = {}

        # 0 is <pad>. userid and itemid start from 1.
        for item,ids in zip(data_arr, range(1, len(data_arr)+1)):
            item2id[item] = ids 
            id2item[ids] = item

        return item2id, id2item


    def convert_to_id(self, data, user2id, item2id):
        """Convert users/items to IDs.

        Args:
            data: Data.
            user2id: Mapping dictionary of users and IDs.
            item2id: Mapping dictionary of items and IDs.

        Returns:
            data: Converted data.
        """

        if 'userId' in data.columns:
            data['userId'] = data['userId'].apply(lambda x: user2id[x])

        data['itemId'] = data['itemId'].apply(lambda x: item2id[x])
        data['item_user_sequence'] = data['item_user_sequence'].apply(lambda seq: list(map(lambda u: user2id[u], seq)))

        return data


    def get_num_user_from_data(self, data):
        """Get total users.

        Args:
            data: Data.

        Returns:
            Total users.
        """

        n_users = set()
        list(map(lambda users: n_users.update(users), data['item_user_sequence'].tolist()))
        return len(n_users)


    def build_dataset(self, dataset, min_seq, test_ratio, batch_size, data_percent):
        """Prepare a dataset for the model, which returns 3 ID files:
            `{xxx}_train.id`: Train set
                with 4 columns (rating, userID, itemID, item historical sequence).
            `{xxx}_test.id`: Test set
                with 4 columns (rating, userID, itemID, item historical sequence).
            `{xxx}_item_user_sequence.id`: Item historical sequence
                with 2 columns (itemID, item historical sequence).

        Args:
            dataset: Dataset name.
            min_seq: Minimum length of sequence.
            test_ratio: Test set splitting ratio.
            batch_size: Batch size.
            data_percent: Percentage of data used to build a dataset.
            If you would like create a dataset from the entire data file, the value is 1.0.
        """

        prefix = '{}_{}_minseq{}_bs{}_dper{}'.format(dataset,
                                                     str(int(100-test_ratio*100))
                                                     + str(int(test_ratio*100)),
                                                     min_seq, batch_size, data_percent)
        print(get_time(), 'Dataset building start!! ...')
        print(get_time(), '\t| Dataset:', dataset)
        print(get_time(), '\t| Data version:', prefix)
        start_time = time()

        # Read data file.
        print(get_time(), '\t| Read dataset ...')
        column_name = ['userId', 'itemId', 'rating', 'timestamp']
        file_path = os.path.join(self.data_path, dataset, '{}_ratings.csv'.format(dataset))
        self.data = pd.read_csv(file_path)
        self.data.columns = column_name
        self.labels.update(self.data['rating'].unique())
        print(get_time(), '\t    | #Record:{} #User:{} #Item:{} #Rating:{} ({})'.format(self.data.shape[0], self.data['userId'].nunique(), self.data['itemId'].nunique(), self.data['rating'].nunique(), sorted(list(self.data['rating'].unique()))))
        
        # Decrease data (if any).
        self.data = self.data.sort_values(by=['timestamp'])
        if data_percent < 1:
            data_size = int(self.data.shape[0]*data_percent)
            self.data = self.data.head(data_size)
            print(get_time(), '\t| Decrease data ...')
            print(get_time(), '\t    | #Record:{} #User:{} #Item:{} #Rating:{} ({})'.format(self.data.shape[0], self.data['userId'].nunique(), self.data['itemId'].nunique(), self.data['rating'].nunique(), sorted(list(self.data['rating'].unique()))))
        print(self.data.sample(5))

        # Create sequence and Remove unfrequent user and item.
        print(get_time(), '\t| Create sequence and Remove unfrequent user and item ...')
        print(get_time(), '\t    | Minimum sequence length:', min_seq)
        print(get_time(), '\t    | creating user-item sequence ...')
        self.ui_seq_df = self.create_sequence('ui')
        print(get_time(), '\t    | finding unfrequent user ...')
        unfreq_user = self.find_unfrequent('user', min_seq)
        print(get_time(), '\t    | removing unfrequent user ...')
        self.ui_seq_df = self.remove_unfrequent(self.ui_seq_df, unfreq_user, 'user')
        self.data = self.remove_unfrequent(self.data, unfreq_user, 'user')

        print(get_time(), '\t    | creating item-user sequence ...')
        self.iu_seq_df = self.create_sequence('iu')
        print(get_time(), '\t    | finding unfrequent item ...')
        unfreq_item = self.find_unfrequent('item', min_seq)
        print(get_time(), '\t    | removing unfrequent item ...')
        print(get_time(), '\t\t| Before: #Record:{}\t#Unfreq_user:{}\t#Unfreq_item:{}\t#User:{}\t#Item:{}'.format(self.data.shape[0], len(unfreq_user), len(unfreq_item), self.data['userId'].nunique(), self.data['itemId'].nunique()))
        self.iu_seq_df = self.remove_unfrequent(self.iu_seq_df, unfreq_item, 'item')
        self.data = self.remove_unfrequent(self.data, unfreq_item, 'item')

        unfreq_user = self.find_unfrequent('user', min_seq)
        unfreq_item = self.find_unfrequent('item', min_seq)
        print(get_time(), '\t\t| After: #Record:{}\t#Unfreq_user:{}\t#Unfreq_item:{}\t\t#User:{}\t#Item:{}'.format(self.data.shape[0], len(unfreq_user), len(unfreq_item), self.data['userId'].nunique(), self.data['itemId'].nunique()))

        print(get_time(), '\t    | Insert sequence to data')
        self.insert_sequence() 
        self.data = self.data[['rating', 'userId', 'itemId', 'item_user_sequence']]
        print(self.data.sample(5))

        # Split train and test set.
        print(get_time(), '\t| Split train/test ...')
        print(get_time(), '\t    | Train ratio:{} Test ratio:{}'.format(1.0-test_ratio, test_ratio))
        self.df_train, self.df_test = train_test_split(self.data, test_size=test_ratio, random_state=32)

        # Prepare for batch size.
        print(get_time(), '\t| Prepare for batch size and Remove unseen data ...')
        print(get_time(), '\t    Before:')
        print(get_time(), '\t    | [Train]\t#Record:{} \t#User:{} \t#Item:{}'.format(self.df_train.shape[0], self.get_num_user_from_data(self.df_train), self.df_train['itemId'].nunique()))
        print(get_time(), '\t    | [Test]\t#Record:{} \t#User:{} \t#Item:{}'.format(self.df_test.shape[0], self.get_num_user_from_data(self.df_test), self.df_test['itemId'].nunique()))
        self.df_train = self.df_train.head((self.df_train.shape[0] // batch_size) * batch_size)

        self.create_vocab(self.df_train, 'user')
        self.create_vocab(self.df_train, 'item')

        # Remove unseen data.
        self.df_test = self.df_test[self.df_test['itemId'].isin(self.items)]
        self.ui_seq_df = self.ui_seq_df[self.ui_seq_df['userId'].isin(self.users)]
        self.iu_seq_df = self.iu_seq_df[self.iu_seq_df['itemId'].isin(self.items)]
        self.df_test = self.df_test.head((self.df_test.shape[0] // batch_size) * batch_size)

        print(get_time(), '\t    After:')
        print(get_time(), '\t    | [Train]\t#Record:{} \t#User:{} \t#Item:{}'.format(self.df_train.shape[0], self.get_num_user_from_data(self.df_train), self.df_train['itemId'].nunique()))
        print(get_time(), '\t    | [Test]\t#Record:{} \t#User:{} \t#Item:{}'.format(self.df_test.shape[0], self.get_num_user_from_data(self.df_test), self.df_test['itemId'].nunique()))
        print(get_time(), '\t    Item-User sequence length: min={} mean={} max={}'.format(self.iu_seq_df['len_item_user_sequence'].min(),
                                                                                int(self.iu_seq_df['len_item_user_sequence'].mean()),
                                                                                self.iu_seq_df['len_item_user_sequence'].max()))

        # Convert to id
        print(get_time(), '\t| Convert to ID ...')
        # Generate ids from only train data
        print(get_time(), '\t    | creating user2id dict ...')
        user2id, id2user = self.generate_dicts(sorted(self.users))
        print(get_time(), '\t    | creating item2id dict ...')
        item2id, id2item = self.generate_dicts(sorted(self.items))

        print(get_time(), '\t    | converting train data ...')
        self.df_train = self.convert_to_id(self.df_train, user2id, item2id)
        print(get_time(), '\t    | converting test data ...')
        self.df_test = self.convert_to_id(self.df_test, user2id, item2id)
        print(get_time(), '\t    | converting item-user sequence data ...')
        self.iu_seq_df = self.convert_to_id(self.iu_seq_df, user2id, item2id)

        print(get_time(), '\t| Sample of train data:')
        print(self.df_train.sample(5))
        print(get_time(), '\t| Sample of test data:')
        print(self.df_test.sample(5))

        # Write to file
        print(get_time(), '\t| Write to file ...')
        print(get_time(), '\t    | writing {} to {}'.format(prefix + '_train.id', self.data_path + dataset))
        self.df_train.to_csv(self.data_path + dataset + '/' + prefix + '_train.id', index=False)
        print(get_time(), '\t    | writing {} to {}'.format(prefix + '_test.id', self.data_path + dataset))
        self.df_test.to_csv(self.data_path + dataset + '/' + prefix + '_test.id', index=False)
        print(get_time(), '\t    | writing {} to {}'.format(prefix + '_item_user_sequence.id', self.data_path + dataset))
        self.iu_seq_df.to_csv(self.data_path + dataset + '/' + prefix + '_item_user_sequence.id', index=False)

        print(get_time(), 'Dataset building finished!! [%.2f s]' %(time()-start_time))


    def load_dataset(self, dataset, data_version):
        """Load dataset into dataframe and print its properties.

        Args:
            dataset: Dataset name.
            data_version: Dataset version.
        """

        print(get_time(), 'Dataset loading start!! ...')
        print(get_time(), '\t| Data Version:', data_version)
        start_time = time()

        columns_name = ['rating', 'userId', 'itemId', 'item_user_sequence']
        print(get_time(), '\t| reading train data ...')
        self.df_train = pd.read_csv(self.data_path + dataset + '/' + data_version + '_train.id',
                                    converters={'item_user_sequence': lambda x: loads(x)})
        self.df_train.columns = columns_name

        print(get_time(), '\t| reading test data ...')
        self.df_test = pd.read_csv(self.data_path + dataset + '/' + data_version + '_test.id',
                                   converters={'item_user_sequence': lambda x: loads(x)})
        self.df_test.columns = columns_name

        self.labels.update(self.df_train['rating'].unique())
        self.labels.update(self.df_test['rating'].unique())

        columns_name = ['itemId', 'item_user_sequence', 'len_item_user_sequence']
        print(get_time(), '\t| reading item-user sequence data ...')
        self.iu_seq_df = pd.read_csv(self.data_path + dataset + '/' + data_version + '_item_user_sequence.id',
                                     converters={'item_user_sequence': lambda x: loads(x)})
        self.iu_seq_df.columns = columns_name

        # Build vocab user and item
        print(get_time(), '\t| loading user vocab ...')
        self.create_vocab(self.df_train, 'user')
        print(get_time(), '\t| loading item vocab ...')
        self.create_vocab(self.df_train, 'item')

        print(get_time(), '\t| Data detail:')
        print(get_time(), '\t    | #Rating:{} ({})'.format(self.get_num_label(), self.labels))
        print(get_time(), '\t    | Train:\t#Record:{} \t#User:{} \t#Item:{}'.format(self.df_train.shape[0], self.get_num_user(), self.get_num_item()))
        print(get_time(), '\t    | Test: \t#Record:{} \t#User:{} \t#Item:{}'.format(self.df_test.shape[0], self.get_num_user_from_data(self.df_test), self.df_test['itemId'].nunique()))
        print(get_time(), '\t    | Item-User sequence length: min={} mean={} max={}'.format(self.iu_seq_df['len_item_user_sequence'].min(),
                                                                                int(self.iu_seq_df['len_item_user_sequence'].mean()),
                                                                                self.iu_seq_df['len_item_user_sequence'].max()))

        print(get_time(), 'Dataset loading finished!! [%.2f s]' %(time()-start_time))


    def get_num_user(self):
        """Get total users in the dataset."""
        return len(self.users)


    def get_num_item(self):
        """Get total items in the dataset."""
        return len(self.items)


    def get_num_label(self):
        """Get total labels in the dataset."""
        return len(self.labels)


if __name__ == "__main__":
    args = parse_args()
    data_path = args.data_path
    dataset = args.dataset
    min_seq = args.min_seq
    test_ratio = args.test_ratio
    batch_size = args.batch_size
    data_percent = args.data_percent
    data_version = args.show_details

    ds = Dataset(data_path)

    if data_version:
        ds.load_dataset(dataset, data_version)
    else:
        ds.build_dataset(dataset, min_seq, test_ratio, batch_size, data_percent)
