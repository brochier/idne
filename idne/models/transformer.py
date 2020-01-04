import tensorflow as tf

def activation_attention(q, k, v, mask, activation):
    batch_size = tf.cast(tf.shape(k)[0], tf.float32)
    d = tf.cast(tf.shape(q)[-1], tf.float32)
    q_tiled = tf.tile(tf.expand_dims(q, 0), [batch_size, 1, 1]) # (batch_size, 1, embedding_size)
    raw_attention_logits = tf.matmul(q_tiled, tf.transpose(k, [0,2,1])) #/ tf.sqrt(d)# + qb_tiles # (batch_size, seq_len_q, seq_len_k)

    # Distribution on the topics
    # (batch_size, num_pools, 1)
    attention_logits = activation(raw_attention_logits+(mask[:, tf.newaxis, :] * -1e9))


    words_weights = tf.reduce_sum(attention_logits, axis=1, keepdims=True)
    zero_mask = tf.cast(tf.math.equal(words_weights, 0), tf.float32)
    words_weights += zero_mask * 1e-9
    attention_weights = attention_logits / words_weights

    """
    words_weights = tf.reduce_sum(tf.reduce_sum(attention_logits, axis = 2, keepdims=True), axis=1, keepdims=True)
    zero_mask = tf.cast(tf.math.equal(words_weights, 0), tf.float32)
    words_weights += zero_mask * 1e-9
    attention_weights = attention_logits / words_weights
    """

    # (batch_size, num_pool, d)
    sums = tf.reduce_sum(1-mask, axis=-1, keepdims=True)
    zero_mask = tf.cast(tf.math.equal(sums, 0), tf.float32)
    sums += zero_mask * 1e-9
    alphas = tf.reduce_sum(attention_weights, axis=-1) / sums
    topic_vectors = tf.matmul(attention_weights, v) / sums[:, tf.newaxis, :]

    output = tf.reduce_sum(topic_vectors, axis=1)
    # (batch_size, 1, d)
    #output = tf.matmul( alphas[:, tf.newaxis, :], output)

    # (batch_size, d)
    #output = tf.reshape(output, [batch_size, d])

    return output, attention_weights, alphas


def cosine_loss(x):
    x = tf.math.l2_normalize(x, axis=-1)
    prod = tf.matmul(x, x, transpose_b=True)
    cos_pos = tf.log(1 + tf.exp(10 * prod - 1))
    loss = tf.norm(cos_pos) - tf.norm(tf.linalg.diag_part(cos_pos))
    return loss