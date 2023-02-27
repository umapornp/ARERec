import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
import tensorflow as tf
tf.config.optimizer.set_jit(True)
from tensorflow.keras import regularizers


class RegionEmbedding(tf.keras.layers.Layer):
    """Region Embedding layer."""

    def __init__(self, re_dropout, name='region_embedding', **kwargs):
        """Constructor for `RegionEmbedding`.

        Args:
            re_dropout: Dropout values.
        """

        super(RegionEmbedding, self).__init__(name=name, **kwargs)
        self.re_dropout = re_dropout
        self.neighbor_profile_dropout = tf.keras.layers.Dropout(self.re_dropout[0])
        self.item_profile_dropout = tf.keras.layers.Dropout(self.re_dropout[1])
        self.neightbor_rating_dropout = tf.keras.layers.Dropout(self.re_dropout[2])
        self.user_rating_vector_dropout = tf.keras.layers.Dropout(self.re_dropout[3])


    def call(self, neighbor_emb, item_emb, K_user_item, K_item_user, weight):
        """Process of `RegionEmbedding`.

        Args:
            neighbor_emb  : Neighbor embedding.     [batch_size, max_seq, emb_size]
            item_emb      : Item embedding.         [batch_size, emb_size]
            K_user_item   : User-Item LCU.          [batch_size, max_seq, region_size, emb_size]
            K_item_user   : Item-User LCU.          [batch_size, max_seq, region_size, emb_size]
            weight        : Mask for padding value. [batch_size, max_seq]

        Returns:
            user_rating_vector : User rating vector. [batch_size, region_size]
        """

        # Perform a dot-product operation between the personalized neighbor embedding
        # and the item-user LCU to generate the user profile.

        # `neighbor_profile` = [batch_size, max_seq, region_size, 1]
        neighbor_profile = tf.matmul(K_item_user, tf.expand_dims(neighbor_emb, -1))

        # `neighbor_profile` = [batch_size, max_seq, region_size]
        neighbor_profile = tf.squeeze(neighbor_profile)
        if self.re_dropout[0]:
            neighbor_profile = self.neighbor_profile_dropout(neighbor_profile)

        # Perform a dot-product operation between the item embedding and the user-item LCU
        # to generate the item profile.
        item_profile = tf.expand_dims(item_emb, -1)          # [batch, emb_size, 1]
        item_profile = tf.expand_dims(item_profile, 1)       # [batch, 1, emb_size, 1]
        item_profile = tf.matmul(K_user_item, item_profile)  # [batch_size, max_seq, region_size, 1]
        item_profile = tf.squeeze(item_profile)              # [batch_size, max_seq, region_size]
        if self.re_dropout[1]:
            item_profile = self.item_profile_dropout(item_profile)

        # Perform an element-wise multiplication between the user profile and the item profile
        # to generate the neighbor rating score, as inspired by NCF technique.
        neighbor_rating = neighbor_profile * item_profile   # [batch_size, max_seq, region_size]
        if self.re_dropout[2]:
            neighbor_rating = self.neightbor_rating_dropout(neighbor_rating)
        weight = tf.expand_dims(weight, -1)         # [batch_size, max_seq, 1]
        neighbor_rating = neighbor_rating * weight  # [batch_size, max_seq, region_size]

        # Perform a max-pooling operation on the rating vectors from every neighbor
        # to extract the most predictive features in the region.
        user_rating_vector = tf.reduce_max(neighbor_rating, axis=1)  # [batch, region_size]
        if self.re_dropout[3]:
            user_rating_vector = self.user_rating_vector_dropout(user_rating_vector)

        return user_rating_vector


class Attention(tf.keras.layers.Layer):
    """Attention layer."""

    def __init__(self, region_size, emb_size, num_heads, regularize, name='Attention', **kwargs):
        """Constructor for `Attention`.

        Args:
            region_size : Size of region.
            emb_size    : Size of embedding.
            num_heads   : Number of attention heads.
            regularize  : Regularization.
        """

        super(Attention, self).__init__(name=name, **kwargs)
        self.region_size = region_size
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.regularize = regularize

        assert emb_size % self.num_heads == 0
        self.depth = emb_size // self.num_heads

        self.wq = tf.keras.layers.Dense(emb_size, name='weight_q',
                                        kernel_regularizer=regularizers.l2(self.regularize[0]))
        self.wk = tf.keras.layers.Dense(emb_size, name='weight_k',
                                        kernel_regularizer=regularizers.l2(self.regularize[1]))
        self.wv = tf.keras.layers.Dense(emb_size, name='weight_v',
                                        kernel_regularizer=regularizers.l2(self.regularize[2]))
        self.dense = tf.keras.layers.Dense(emb_size, name='weight_concat',
                                           kernel_regularizer=regularizers.l2(self.regularize[3]))


    def split_heads(self, x, batch_size, isQ=False):
        """Split the last dimension into (num_heads, depth)
        and transpose the result such that the shape is (batch_size, num_heads, max_seq, depth).

        Args:
            x: Vector.
            batch_size: Batch size.
            isQ: Whether `x` vector is Q.

        Returns:
            x: Splitted vector.
        """

        if isQ:
            # `x` = [batch_size, num_heads, depth]
            x = tf.reshape(x, (batch_size, self.num_heads, self.depth))
        else:
            # `x` = [batch_size, max_seq, num_heads, depth]
            x = tf.reshape(x, (batch_size, x.shape[1], self.num_heads, self.depth))

            # `x` = # [batch_size, num_heads, max_seq, depth]
            x = tf.transpose(x, perm=[0, 2, 1, 3])      

        return x


    def scaled_dot_product_attention(self, q, k, v, mask=None):
        """Calculate the attention weights.
        q, k, v must have matching leading dimensions.
        k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
        The mask has different shapes depending on its type(padding or look ahead)
        but it must be broadcastable for addition.

        Args:
            q: Query.    [batch_size, num_heads, depth]
            k: Key.      [batch_size, num_heads, max_seq, depth]
            v: Value.    [batch_size, num_heads, max_seq, depth]
            mask: The mask for padding value.

        Returns:
            output: Scaled dot product output. [batch_size, num_heads, max_seq, depth]
            attention_weights: Attention weights.
        """

        q = tf.expand_dims(q, 2)                        # [batch_size, num_heads, 1, depth]
        matmul_qk = tf.matmul(q, k, transpose_b=True)   # [batch_size, num_heads, 1, max_seq]
        matmul_qk = tf.squeeze(matmul_qk, axis=2)       # [batch_size, num_heads, max_seq]

        # Scale matmul_qk.
        dk = tf.cast(tf.shape(k)[-1], tf.float32)

        # `scaled_attention_logits` = [batch_size, num_heads, max_seq]
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        # Add the mask to the scaled tensor.
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        # Softmax is normalized on the last axis so that the scores
        # add up to 1.
        # `attention_weights` = [batch_size, num_heads, max_seq]
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

        # `attention_weights` = # [batch_size, num_heads, max_seq, 1]
        attention_weights = tf.expand_dims(attention_weights, -1)

        # `output` = [batch_size, num_heads, max_seq, depth]
        output = attention_weights * v

        return output, attention_weights


    def call(self, q, k, v, mask=None):
        """Process of `Attention`.

        Args:  
            q: Query.  [batch_size, emb_size]
            k: Key.    [batch_size, max_seq, emb_size]
            v: Value.  [batch_size, max_seq, emb_size]
            mask: The mask for padding value.

        Returns:
            output            : Attention output. [batch_size, region_size, emb_size]
            attention_weights : Attention weights.
        """

        batch_size = tf.shape(q)[0]
        q = self.wq(q)  # [batch_size, emb_size]
        k = self.wk(k)  # [batch_size, max_seq, emb_size]
        v = self.wv(v)  # [batch_size, max_seq, emb_size]

        q = self.split_heads(q, batch_size, isQ=True) # [batch_size, num_heads, depth]
        k = self.split_heads(k, batch_size)           # [batch_size, num_heads, max_seq, depth]
        v = self.split_heads(v, batch_size)           # [batch_size, num_heads, max_seq, depth]

        # `scaled_attention` = [batch_size, num_heads, max_seq, depth]
        scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v, mask)

        # `scaled_attention` = [batch_size, max_seq, num_heads, depth]
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        # `concat_attention` = [batch_size, max_seq, emb_size]
        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, tf.shape(scaled_attention)[1], self.emb_size))

        # `output` = [batch_size, max_seq, emb_size]
        output = self.dense(concat_attention)

        return output, attention_weights


class ARERec(tf.keras.Model):
    """ARERec model."""

    def __init__(self,
                 region_size,
                 user_size,
                 item_size,
                 emb_size,
                 num_heads,
                 num_class,
                 batch_size,
                 all_seq,
                 dropout,
                 regularize,
                 use_attention=False,
                 name='ARERec', **kwargs):
        """Constructor for `ARERec`.

        Args:
            region_size   : Size of region.
            user_size     : Number of users.
            item_size     : Number of items.
            emb_size      : Size of embedding.
            num_heads     : Number of attention heads.
            num_class     : Number of rating classes.
            batch_size    : Batch size.
            all_seq       : Number of User-Item LCUs and Item-User LCUs.
            dropout       : Dropout.
            regularize    : Regularization.
            use_attention : Whether to use `Attention` layer.
        """

        super(ARERec, self).__init__(name=name, **kwargs)
        self.region_size = region_size
        self.batch_size = batch_size
        self.use_attention = use_attention

        # Initialize user embedding. [user_size, emb_size]
        self.user_embedding = tf.keras.layers.Embedding(user_size,
                                                        emb_size,
                                                        mask_zero=True,
                                                        trainable=True,
                                                        name='user_embedding')

        # Initialize item embedding. [item_size, emb_size]
        self.item_embedding = tf.keras.layers.Embedding(item_size,
                                                        emb_size,
                                                        input_length=1,
                                                        mask_zero=True,
                                                        trainable=True,
                                                        name='item_embedding')

        # Initialize user-item local context unit (LCU). [all_seq, region_size, emb_size]
        user_item_LCU_init = tf.random_normal_initializer(mean=0.0, stddev=0.01)
        self.user_item_LCU = tf.Variable(user_item_LCU_init(shape=[all_seq, region_size, emb_size]),
                                         trainable=True,
                                         name='K_user_item')

        # Initialize item-user local context unit (LCU). [all_seq, region_size, emb_size]
        item_user_LCU_init = tf.random_normal_initializer(mean=0.0, stddev=0.01)
        self.item_user_LCU = tf.Variable(item_user_LCU_init(shape=[all_seq, region_size, emb_size]),
                                         trainable=True,
                                         name='K_item_user')

        # Set dropout (if any).
        self.dropout = dropout
        self.user_dropout = tf.keras.layers.Dropout(self.dropout[0])
        self.neighbor_dropout = tf.keras.layers.Dropout(self.dropout[1])
        self.item_dropout = tf.keras.layers.Dropout(self.dropout[2])
        self.k_user_item_dropout = tf.keras.layers.Dropout(self.dropout[3])
        self.k_item_user_dropout = tf.keras.layers.Dropout(self.dropout[4])

        # Set regularization (if any).
        self.regularize = regularize

        # Create `Attention` layer.
        if self.use_attention:
            self.attention = Attention(region_size, emb_size, num_heads, self.regularize[:4])

        # Create `RegionEmbedding` layer.
        self.region_embedding = RegionEmbedding(self.dropout[5:])

        # Create fully connected layer.
        self.fc = tf.keras.layers.Dense(num_class,
                                        activation='softmax',
                                        name='fc',
                                        kernel_regularizer=regularizers.l2(self.regularize[4]))


    def create_padding_mask(self, sequence):
        """Create mask for padding value.

        Args:
            sequence: Sequence.

        Returns:
            weight: Mask for padding value. [batch_size, seq_len]
        """

        def mask(x):
            return tf.cast(tf.greater(tf.cast(x, tf.int32), tf.constant(0)), tf.float32)

        # `weight` = [batch_size, seq_len]
        weight = tf.map_fn(mask, sequence, dtype=tf.float32,
                           parallel_iterations=50, swap_memory=True)

        return weight


    @tf.function(experimental_compile=True)
    def call(self, inputs):
        """Process of `ARERec`.

        Args:
            inputs: Input data which is tuple of
                user: Target users. [batch_size]
                item: Target items. [batch_size]
                item_user_sequence: Item historical sequence that contain
                    list of users who rate the item (i.e., neighbors). [batch_size, max_seq]
                item_user_sequence_index: LCU IDs. [batch_size, max_seq]

        Returns:
            rating: Predicted ratings. [batch_size, num_class]
        """

        # Set input.
        user = inputs[0]                      # [batch_size]
        item = inputs[1]                      # [batch_size]
        item_user_sequence = inputs[2]        # [batch_size, max_seq]
        item_user_sequence_index = inputs[3]  # [batch_size, max_seq]

        # Perform an embedding lookup on the neighbor IDs to obtain neighbor embedding.
        # `neighbor_emb` = [batch_size, max_seq, emb_size]
        neighbor_emb = self.user_embedding(item_user_sequence)
        if self.dropout[1] and not self.use_attention:
            neighbor_emb = self.neighbor_dropout(neighbor_emb)

        # Perform an embedding lookup on the item IDs to obtain item embedding.
        item_emb = self.item_embedding(item)  # [batch_size, emb_size]
        if self.dropout[2]:
            item_emb = self.item_dropout(item_emb)

        # Perform an embedding lookup on the LCU IDs to obtain User-Item LCU.
        # `K_user_item` = [batch_size, max_seq, region_size, emb_size]
        K_user_item = tf.nn.embedding_lookup(self.user_item_LCU, item_user_sequence_index)
        if self.dropout[3]:
            K_user_item = self.k_user_item_dropout(K_user_item)

        # Perform an embedding lookup on the LCU IDs to obtain Item-User LCU.
        # `K_item_user` = [batch_size, max_seq, region_size, emb_size]
        K_item_user = tf.nn.embedding_lookup(self.item_user_LCU, item_user_sequence_index)
        if self.dropout[4]:
            K_item_user = self.k_item_user_dropout(K_item_user)

        # Apply `Attention` layer.
        if self.use_attention:
            # Perform an embedding lookup on the user IDs to obtain user embedding.
            user_emb = self.user_embedding(user)    # [batch_size, emb_size]
            if self.dropout[0]:
                user_emb = self.user_dropout(user_emb)

            mask_seq = tf.cast(tf.math.equal(item_user_sequence, 0), tf.float32)
            mask_seq = tf.expand_dims(mask_seq, axis=1)

            # Obtain neighbor embeddings that are weighted based on
            # their relevance to the target user.
            # `neighbor_emb` = [batch_size, max_seq, emb_size]
            neighbor_emb, attention_weight = self.attention(q=user_emb,
                                                            k=neighbor_emb,
                                                            v=neighbor_emb,
                                                            mask=mask_seq)
            if self.dropout[1]:
                neighbor_emb = self.neighbor_dropout(neighbor_emb)

        # `weight` = [batch_size, seq_len]
        weight = self.create_padding_mask(item_user_sequence)

        # Apply `RegionEmbedding` layer.
        # `rating_vector` = [batch_size, region_size]
        rating_vector = self.region_embedding(neighbor_emb=neighbor_emb,
                                              item_emb=item_emb,
                                              K_user_item=K_user_item,
                                              K_item_user=K_item_user,
                                              weight=weight)

        # Apply Fully Connected layer.
        # `rating` = [batch_size, num_class]
        rating = self.fc(rating_vector)

        return rating
