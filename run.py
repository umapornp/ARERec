from dataset import Dataset 
from preprocess import Preprocess
from model import ARERec
from evaluate import Evaluate

import os
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
logging.getLogger('tensorflow').disabled = True

import re
import sys
import csv
import argparse
import numpy as np
import tensorflow as tf
tf.config.optimizer.set_jit(True)
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.losses import categorical_crossentropy
from datetime import datetime
from time import time


#################### Arguments ####################
def parse_args():
    """Arguments for running the model via command-line interfaces."""
    
    parser = argparse.ArgumentParser(description="Run Model")
    parser.add_argument('--data_path', nargs='?', default='Data/',
                        help='Data path')
    parser.add_argument('--dataset', nargs='?', default='ml-1m',
                        help='Dataset name')
    parser.add_argument('--data_version', nargs='?', default='',
                        help='Data version')
    parser.add_argument('--model_path', nargs='?', default='Model/',
                        help='Model path')
    parser.add_argument('--model_name', nargs='?', default='',
                        help='Model name')
    parser.add_argument('--save_model', type=int, default=1,
                        help='Whether to save the model')
    parser.add_argument('--load_model', type=int, default=0,
                        help='Whether to load the model')
    parser.add_argument('--load_weight_path', nargs='?', default='',
                        help='Weight path')
    parser.add_argument('--do_train', type=int, default=1,
                        help='Whether to train model')
    parser.add_argument('--do_eval', type=int, default=1,
                        help='Whether to evaluate model')
    parser.add_argument('--save_checkpoints_epochs', type=int, default=2,
                        help='How often to save the model checkpoint')
    parser.add_argument('--epochs', type=int, default=2,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--region_size', type=int, default=7,
                        help='Region size of model')
    parser.add_argument('--emb_size', type=int, default=64,
                        help='Embedding size of model')
    parser.add_argument('--use_attention', type=int, default=1,
                        help='Whether to use attention')
    parser.add_argument('--num_head', type=int, default=2,
                        help='Attention Head')
    parser.add_argument('--max_seq_mode', nargs='?', default='mean',
                        help='Max of sequence lenght mode (max or min)')
    parser.add_argument('--max_seq', type=int, default=-1,
                        help='Max of sequence lenght')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--optimizer', nargs='?', default='adam',
                        help='Specify an optimizer: adam, rms')
    parser.add_argument('--use_dropout', type=int, default=0,
                        help='Whether to use dropout')
    parser.add_argument('--dropout', nargs='?', default=[0,0,0,0.5,0.5,0,0,0,0],
                        help='Dropout rate [user_emb, neighbor_emb, item_emb, K_user_item, K-item_user, neighbor_profile, item_profile, neighbor_rating, user_rating]')
    parser.add_argument('--use_regularizer', type=int, default=0,
                        help='Whether to use regularizer')
    parser.add_argument('--regularizer', nargs='?', default=[0.001,0.001,0.001,0.001,0.001],
                        help='L2 regularizer rate [q,k,v,concat,fc]')
    parser.add_argument('--topk', nargs='?', default=[3,5,7,10],
                        help='Top-K evaluation.')
    parser.add_argument('--hrcutoff', type=int, default=3,
                        help='HR cutoff threshold evaluation.')
    
    return parser.parse_args()


def get_time():
    """Get the current time."""
    return '[{}]:>\t'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')) 


class EvaluateModel(tf.keras.callbacks.Callback):
    """Callbacks during model training."""

    def __init__(self, id2label, batch_size, filename, topk=[10], hrcutoff=3):
        """Constructor for `EvaluateModel`.

        Args:
            id2label: Mapping dictionary of IDs and its label.
            batch_size: Batch size.
            filename: Name of result file.
            topk: Top-k for evaluation metrics.
            hrcutoff: Cufoff rating of HitRate.
        """

        super(EvaluateModel, self).__init__()
        self.id2label = id2label
        self.batch_size = batch_size
        self.topk = topk
        self.hrcutoff = hrcutoff
        self.best_result_at_k = {}

        self.filename = filename
        self.writer = None
        self.csvfile = None
        self.write_header = True


    def on_train_begin(self, logs=None):
        """Processes when training begin."""

        # Set a timer for training.
        self.start_train_time = time()

        # Prepare to save the result file.
        if os.path.exists(self.filename):
            self.write_header = False

        print('==============================================================================================================', file=sys.stderr)
        print(get_time(), 'Training Start!...', file=sys.stderr)


    def on_epoch_begin(self, epoch, logs=None):
        """Processes when epoch begin."""

        # Set a timer for epoch.
        self.start_epoch_time = time()


    def on_epoch_end(self, epoch, logs=None):
        """Processes when epoch end."""

        # Evaluate the model.
        result_at_k = {'hr':{}, 'ndcg':{}}
        eval_model = Evaluate()
        hitrate_print = ''
        ndcg_print = ''
        for k in self.topk:
            hitrate, ndcg = eval_model.get_evaluate(self.model, test_x, test_y,
                                                    self.id2label, self.batch_size,
                                                    k, self.hrcutoff)
            result_at_k['hr'][k] = hitrate
            result_at_k['ndcg'][k] = ndcg

            # Find the best NDCG.
            if k not in self.best_result_at_k:
                self.best_result_at_k[k] = {'epoch':-1, 'loss':-1, 'val_loss':-1, 'hr':-1, 'ndcg':-1}
            if ndcg > self.best_result_at_k[k]['ndcg']:
                self.best_result_at_k[k]['epoch'] = epoch+1
                self.best_result_at_k[k]['loss'] = logs['loss']
                self.best_result_at_k[k]['val_loss'] = logs['val_loss']
                self.best_result_at_k[k]['hr'] = hitrate
                self.best_result_at_k[k]['ndcg'] = ndcg

            # Prepare to print the result.
            hitrate_print += '\t|HR@{}={:.6f}\t'.format(k, hitrate)
            ndcg_print += '\t|NDCG@{}={:.6f}'.format(k, ndcg)

        # Log to CSV file.
        fields = ['epoch', 'loss', 'val_loss']
        list(map(lambda x: fields.append('hr@{}'.format(x)), topk))
        list(map(lambda x: fields.append('ndcg@{}'.format(x)), topk))
        data = [epoch+1, logs['loss'], logs['val_loss']]
        list(map(lambda x: data.append(x), result_at_k['hr'].values()))
        list(map(lambda x: data.append(x), result_at_k['ndcg'].values()))
        row_dict = [{i[0]:i[1] for i in zip(fields, data)}]
        with open(self.filename, 'a', newline='') as self.csvfile:
            self.writer = csv.DictWriter(self.csvfile, fieldnames=fields)
            if self.write_header:
                self.writer.writeheader()
            self.writer.writerows(row_dict)
            self.csvfile.close()
            self.writer = None

        # Print the result.
        print('Epoch {:05d}:'.format(epoch+1))
        print('\t|loss:{:.6f} \t\t|val_loss:{:.6f} '.format(logs['loss'], logs['val_loss']))
        print(hitrate_print)
        print(ndcg_print)
        print('Epoch {:05d}: Finished. [{:.2f} s]'.format(epoch+1, time()-self.start_epoch_time))
        print('--------------------------------------------------------------------------------------------------------------', file=sys.stderr)

        return


    def on_train_end(self, logs=None):
        """Processes when training end."""

        # Print the result.
        print (get_time(), 'Training Done! [{:.2f} s]'.format(time()-self.start_train_time), file=sys.stderr)
        for k in self.topk:
            print('Best Epoch(k={}) \tis {:05d} \t|loss:{:.6f} \t|val_loss:{:.6f} \t|HR:{:.6f} \t|NDCG:{:.6f}' \
            .format(k, self.best_result_at_k[k]['epoch'], \
                    self.best_result_at_k[k]['loss'], self.best_result_at_k[k]['val_loss'], \
                    self.best_result_at_k[k]['hr'], self.best_result_at_k[k]['ndcg']))
        return


def get_num_user_from_data(data):
    """Get the number of users in the data.

    Args:
        data: Data.

    Returns:
        Number of users in the data.
    """

    n_users = set()
    data['item_user_sequence'].apply(lambda x: n_users.update(x))
    return len(n_users)


if __name__ == '__main__':
    args = parse_args()

    # Hyperparameters
    region_size = args.region_size
    emb_size = args.emb_size
    use_attention = args.use_attention
    num_head = args.num_head
    batch_size = args.batch_size
    optimizer = args.optimizer
    lr = args.lr
    max_seq_mode = args.max_seq_mode
    max_seq = args.max_seq
    use_dropout = args.use_dropout
    if isinstance(args.dropout, str):
        dropout = eval(args.dropout)
    else:
        dropout = args.dropout
    use_regularizer = args.use_regularizer
    if isinstance(args.regularizer, str):
        regularizer = eval(args.regularizer)
    else:
        regularizer = args.regularizer

    # Data
    data_path = args.data_path
    dataset = args.dataset
    data_version = args.data_version

    # Train
    num_epochs = args.epochs
    initial_epoch = 0
    do_train = args.do_train
    model_path = args.model_path
    model_name = args.model_name
    save_model = args.save_model
    load_model = args.load_model
    load_weight_path = args.load_weight_path
    save_checkpoints_epochs = args.save_checkpoints_epochs

    # Evaluate
    do_eval = args.do_eval
    if isinstance(args.topk, str):
        topk = eval(args.topk)
    else:
        topk = args.topk
    hrcutoff = args.hrcutoff

    # Load dataset.
    print(get_time(), 'Load dataset ...')
    print(get_time(), '\t| Load dataset from', data_version)
    ds = Dataset(data_path)
    ds.load_dataset(dataset, data_version)
    df_train = ds.df_train
    df_test = ds.df_test
    iu_seq_df = ds.iu_seq_df
    labels = ds.labels
    num_user = ds.get_num_user() + 1 # 1 is for paddding
    num_item = ds.get_num_item() + 1
    num_label = ds.get_num_label()
    data_params = data_version.split('_')
    split_ratio = eval(data_params[1])
    data_percent = eval(data_params[4][4:])

    # Set the maximum length of all sequences in the dataset.
    if max_seq > 0:
        print(get_time(), 'Set max sequence length to {}'.format(max_seq))
    elif max_seq_mode:
        if max_seq_mode == 'mean':
            max_seq = int(iu_seq_df['len_item_user_sequence'].mean())
        elif max_seq_mode == 'max':
            max_seq = int(iu_seq_df['len_item_user_sequence'].max())
        print(get_time(), 'Set max sequence length to {} ({})'.format(max_seq, max_seq_mode))

    # Load or resume a previous saved model.
    if model_name and load_model:
        # Restore hyperparameter from a previous saved model.
        model_params = model_name.split('_')
        dataset = model_params[0]
        split_ratio = eval(model_params[1])
        batch_size = eval(model_params[2][2:])
        lr = eval(model_params[3][2:])
        emb_size = eval(model_params[4][3:])
        region_size = eval(model_params[5][3:])
        max_seq = eval(model_params[6][2:])
        optimizer = model_params[7][2:]
        if use_attention:
            num_head = eval(model_params[8][1:])
 
    # Set the filename for saving the model.
    # Naming convention of model name is
    # {dataset}_{split ratio}_bs{batch size}_lr{learning rate}_emb{embedding size}_reg{region size}_ms{max seq}_op{optimizer}.
    # For example, ml-1m_8020_bs256_lr0.0001_emb32_reg7_ms327_opadam_h2
    else:
        model_name = '{}_{}_bs{}_lr{}_emb{}_reg{}_ms{}_op{}'.format(dataset, split_ratio, batch_size, lr, emb_size, region_size, max_seq, optimizer)
        if use_attention:
            model_name = model_name + '_h{}'.format(num_head)

        if use_dropout:
            model_name = model_name + '_do'
        else:
            dropout = [0,0,0,0,0,0,0,0,0]

        if use_regularizer:
            model_name = model_name + '_rz'
        else:
            regularizer = [0,0,0,0,0]

    print(get_time(), 'Model name:', model_name)

    # Create a checkpoint path.
    checkpoint_path = os.path.join(model_path, dataset, model_name)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    # Create a weight path.
    save_weight_path = os.path.join(checkpoint_path, 'weight/cp-{epoch:04d}.ckpt')

    # Create a log path.
    tensorboard_path = os.path.join(checkpoint_path, 'logs_{}'.format(model_name))

    # Create a result path.
    result_path = os.path.join(checkpoint_path, 'result_{}'.format(model_name))


    # Preprocess the dataset.
    print(get_time(), 'Preprocess dataset ...')
    prep = Preprocess()
    iu_seq_df = iu_seq_df.sort_values(by=['itemId'])
    sequence_index_dict, num_lcu = prep.generate_sequence_index(iu_seq_df['itemId'].values, iu_seq_df['item_user_sequence'].values, max_seq)
    label2id, id2label = prep.generate_label_dict(sorted(labels))
    train_x, train_y = prep.preprocess(df_train, label2id, sequence_index_dict, max_seq)
    test_x, test_y = prep.preprocess(df_test, label2id, sequence_index_dict, max_seq)

    num_lcu += 1 # 1 for padding
    print(get_time(), '\t| #User:{} #Item:{} #LCU:{}'.format(num_user, num_item, num_lcu))

    # Build the model.
    print(get_time(), 'Build model ...')
    mymodel = ARERec(region_size, num_user, num_item, emb_size, num_head, num_label, batch_size, num_lcu, dropout, regularizer, use_attention=use_attention)

    # Complie the model.
    print(get_time(), 'Compile model ...')
    if optimizer.lower() == 'rms':
        mymodel.compile(optimizer=RMSprop(learning_rate=lr), loss='categorical_crossentropy')
    elif optimizer.lower() == 'adam':
        mymodel.compile(optimizer=Adam(learning_rate=lr), loss='categorical_crossentropy')

    # Create callbacks.
    evm = EvaluateModel(id2label, batch_size, result_path, topk, hrcutoff)
    tsb = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_path, histogram_freq=1, write_images=True)
    callbacks = [evm, tsb]
    
    # Save the model.
    if save_model:
        ckpt = tf.keras.callbacks.ModelCheckpoint(filepath=save_weight_path,
                                                    save_best_only=False,
                                                    monitor='val_loss',
                                                    save_weights_only=True,
                                                    verbose=1,
                                                    save_freq='epoch',
                                                    period=save_checkpoints_epochs)
        callbacks.insert(0, ckpt)
        print(get_time(), 'Model will checkpoint to', checkpoint_path)


    # Load the model.
    if load_model:
        print(get_time(), 'Load previous model ...')
        initial_epoch = int(re.findall(r'\d+', load_weight_path)[0])
        load_weight_path = os.path.join(checkpoint_path, 'weight', load_weight_path)
        mymodel.load_weights(load_weight_path)
        print(get_time(), '\t| Model load weight from', load_weight_path)
        print(get_time(), '\t| Model resume from Epoch', initial_epoch)


    # Train the model.
    if do_train:
        history = mymodel.fit(train_x, train_y,
                            epochs=num_epochs,
                            initial_epoch = initial_epoch,
                            batch_size=batch_size,
                            callbacks=callbacks,
                            verbose=1,
                            validation_data=(test_x, test_y))

    # Evaluate the model.
    if do_eval and not do_train:
        print(get_time(), 'Evaluate model ...')
        print(get_time(), '\tModel name:', model_name)

        start_epoch_time = time()
        eval_model = Evaluate()
        hitrate_print = ''
        ndcg_print = ''

        for k in topk:
            hitrate, ndcg = eval_model.get_evaluate(mymodel, test_x, test_y, 
                                                        id2label, batch_size, 
                                                        k, hrcutoff)

            hitrate_print += '\t|HR@{}={:.6f}\t'.format(k, hitrate)
            ndcg_print += '\t|NDCG@{}={:.6f}'.format(k, ndcg)
            
        test_loss = mymodel.evaluate(test_x, test_y, batch_size=batch_size, verbose=0)
        
        print(get_time(), '\t|loss:{:.6f}'.format(test_loss))
        print(get_time(), hitrate_print)
        print(get_time(), ndcg_print)
        print(get_time(), 'Evaluaiton Finished. [{:.2f} s]'.format(time()-start_epoch_time))
