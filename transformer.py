import tensorflow as tf
import numpy as np

def create_masks(inp, tar):
    # Encoder padding mask
    enc_padding_mask = create_padding_mask(inp)

    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    dec_padding_mask = create_padding_mask(inp)

    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by
    # the decoder.
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    look_ahead_mask = tf.expand_dims(look_ahead_mask, 0)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask

def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)

def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)

def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable
          to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
    output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e3)

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)
    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights

def split_heads(x, batch_size, num_heads, depth ):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, num_heads, depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
def MultiHeadAttention(d_model, num_heads, q, k, v, mask):
    assert d_model % num_heads == 0
    depth = d_model // num_heads

    wv = tf.keras.layers.Dense(d_model)

    dense = tf.keras.layers.Dense(d_model)
    
    batch_size = tf.shape(q)[0]

    q = tf.keras.layers.Dense(d_model)(q)
    k = tf.keras.layers.Dense(d_model)(k)
    v = tf.keras.layers.Dense(d_model)(v)  

    q = split_heads(q, batch_size, num_heads, depth)  
    k = split_heads(k, batch_size, num_heads, depth)  
    v = split_heads(v, batch_size, num_heads, depth) 

    scaled_attention, attention_weights = scaled_dot_product_attention(
        q, k, v, mask)

    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  
    concat_attention = tf.reshape(scaled_attention,
                                  (batch_size, -1, d_model)) 

    output = tf.keras.layers.Dense(d_model)(concat_attention)

    return output

def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])

def EncoderLayer(d_model, num_heads, dff, x, mask, rate=0.1):
    attn_output = MultiHeadAttention(d_model, num_heads,x, x, x, mask)  
    attn_output = tf.keras.layers.Dropout(rate)(attn_output)
    out1 = tf.keras.layers.BatchNormalization()(x + attn_output)  

    ffn_output = point_wise_feed_forward_network(d_model, dff)(out1) 
    ffn_output = tf.keras.layers.Dropout(rate)(ffn_output)
    out2 = tf.keras.layers.BatchNormalization()(out1 + ffn_output)

    return out2


def DecoderLayer(x, enc_output, look_ahead_mask, padding_mask, d_model, num_heads, dff, rate = 0.1):
    attn1 = MultiHeadAttention(d_model, num_heads,x, x, x, look_ahead_mask)
    attn1 = tf.keras.layers.Dropout(rate)(attn1)
    out1 = tf.keras.layers.BatchNormalization()(attn1 + x)
    attn2 = MultiHeadAttention(d_model, num_heads, out1, enc_output, enc_output, padding_mask)

    attn2 = tf.keras.layers.Dropout(rate)(attn2)
    out2 = tf.keras.layers.BatchNormalization()(attn2 + out1)  
    ffn_output = point_wise_feed_forward_network(d_model, dff)(out2) 
    ffn_output = tf.keras.layers.Dropout(rate)(ffn_output)
    out3 = tf.keras.layers.BatchNormalization()(ffn_output + out2)  
    return out3


def Transformer(input_vocab_size, target_vocab_size, eng_length, fre_length, batch_size = 1 , num_layers = 2):
    d_model = 128
    dff = 512
    num_heads = 1
    dropout_rate = 0.1
    pe_input = 1000
    pe_target = 1000
   
    enc_padding_mask, combined_mask, dec_padding_mask = None, None, None
    input = tf.keras.layers.Input(shape=(eng_length), batch_size = batch_size, dtype = tf.int32)
    target = tf.keras.layers.Input(shape=(fre_length - 1), batch_size = batch_size, dtype = tf.int32)

    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(input, target)


    ## Encoder
    seq_len = tf.shape(input)[1]
    x = tf.keras.layers.Embedding(input_vocab_size, d_model)(input)
    x *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    number = positional_encoding(pe_input, d_model)[:, :seq_len, :]
    x += number
    x = tf.keras.layers.Dropout(dropout_rate)(x)

    for i in range(num_layers):
        x = EncoderLayer(d_model, num_heads, dff, x, enc_padding_mask)
    enc_output = x

    ## Decoder
    seq_len = tf.shape(target)[1]
    attention_weights = {}
    x = tf.keras.layers.Embedding(target_vocab_size, d_model)(target)
    x *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    number = positional_encoding(pe_target, d_model)[:, :seq_len, :]
    x += number
    x = tf.keras.layers.Dropout(dropout_rate)(x)

    for i in range(num_layers):
        x = DecoderLayer(x, enc_output, combined_mask, dec_padding_mask, d_model, num_heads, dff)

    x = tf.keras.layers.Dense(target_vocab_size)(x)
    model = tf.keras.models.Model(inputs=[input, target], outputs=x)
    return model