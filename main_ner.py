import tensorflow as tf
import sys
from gensim.models import Word2Vec
import numpy as np
from collections import Counter
from scipy.linalg import toeplitz
from gensim.models import KeyedVectors


def assemble_model(init_vectors, seq_len, n_tags, out_tag, lr=0.001, train_embeddings=False):
    voc_size = init_vectors.shape[0]
    emb_dim = init_vectors.shape[1]

    d_win = 5
    h1_size = 500
    h2_size = 200

    h1_kernel_shape = (d_win, emb_dim)

    tf_words = tf.placeholder(shape=(None, seq_len), dtype=tf.int32, name="words")
    tf_labels = tf.placeholder(shape=(None, seq_len), dtype=tf.int32, name="labels")
    tf_lengths = tf.placeholder(shape=(None,), dtype=tf.int32, name="lengths")

    def create_embedding_matrix():
        n_dims = init_vectors.shape[1]
        embs = tf.get_variable("embeddings", initializer=init_vectors, dtype=tf.float32, trainable=train_embeddings)
        pad = tf.zeros(shape=(1, n_dims), name="pad")
        emb_matr = tf.concat([embs, pad], axis=0)
        return emb_matr

    def convolutional_layer(input, units, cnn_kernel_shape, activation=None):
        padded = tf.pad(input, tf.constant([[0, 0], [2, 2], [0, 0]]))
        emb_sent_exp = tf.expand_dims(padded, axis=3)
        convolve = tf.layers.conv2d(emb_sent_exp,
                                    units,
                                    cnn_kernel_shape,
                                    activation=activation,
                                    data_format='channels_last',
                                    name="conv_h1")
        return tf.reshape(convolve, shape=(-1, convolve.shape[1], units))

    emv_matr = create_embedding_matrix()

    emb_sent = tf.nn.embedding_lookup(emv_matr, tf_words)

    conv_h1 = convolutional_layer(emb_sent, h1_size, h1_kernel_shape, tf.nn.tanh)

    token_features_1 = tf.reshape(conv_h1, shape=(-1, h1_size))

    local_h2 = tf.layers.dense(token_features_1,
                               h2_size,
                               activation=tf.nn.tanh,
                               name="dense_h2")

    tag_logits = tf.layers.dense(local_h2, n_tags, activation=None)
    logits = tf.reshape(tag_logits, (-1, seq_len, n_tags))

    with tf.variable_scope('loss') as l:
        log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(logits, tf_labels, tf_lengths)
        loss = tf.reduce_mean(-log_likelihood)

    train = tf.train.AdamOptimizer(lr).minimize(loss)

    mask = tf.sequence_mask(tf_lengths, seq_len)
    true_labels = tf.math.equal(tf.boolean_mask(tf_labels, mask), out_tag)
    argmax = tf.math.argmax(logits, axis=-1)
    estimated_labels = tf.math.equal(tf.cast(tf.boolean_mask(argmax, mask), tf.int32), out_tag)
    _,accuracy = tf.contrib.metrics.f1_score(tf.cast(estimated_labels, dtype=tf.int32), tf.cast(true_labels, dtype=tf.int32))

    return {
        'words': tf_words,
        'labels': tf_labels,
        'lengths': tf_lengths,
        'loss': loss,
        'train': train,
        'accuracy': accuracy,
        'argmax': argmax
    }



# sent_flattened = tf.maximum(local_sent_enh, axis=1)
def read_data(path):
    sents_w = [];
    sent_w = []
    sents_t = [];
    sent_t = []

    tags = set()

    with open(path, "r") as conll:
        for line in conll.read().strip().split("\n"):
            if line == '':
                sents_w.append(sent_w)
                sent_w = []
                sents_t.append(sent_t)
                sent_t = []
            else:
                try:
                    word, tag = line.split()
                except:
                    continue
                tags.add(tag)
                sent_w.append(word.lower())
                sent_t.append(tag)

    tags = list(tags);
    tags.sort()

    tagmap = dict(zip(tags, range(len(tags))))
    return sents_w, sents_t, tags, tagmap


def load_model(model_p):
    # model = Word2Vec.load(model_p)
    model = KeyedVectors.load_word2vec_format(model_p)
    voc_len = len(model.vocab)

    vectors = np.zeros((voc_len, 300), dtype=np.float32)

    w2i = dict()

    for ind, word in enumerate(model.vocab.keys()):
        w2i[word] = ind
        vectors[ind, :] = model[word]

    # w2i["*P*"] = len(w2i)

    return model, w2i, vectors


def create_batches(batch_size, seq_len, sents, tags, wordmap, tagmap):
    pad_id = len(wordmap)
    n_sents = len(sents)

    b_sents = []
    b_tags = []
    b_lens = []

    for ind, s in enumerate(sents):
        blank_s = np.ones((seq_len,), dtype=np.int32) * pad_id
        blank_t = np.zeros((seq_len,), dtype=np.int32)

        int_sent = np.array([wordmap.get(w, pad_id) for w in s], dtype=np.int32)
        int_tags = np.array([tagmap.get(t, 0) for t in tags[ind]], dtype=np.int32)

        blank_s[0:min(int_sent.size, seq_len)] = int_sent[0:min(int_sent.size, seq_len)]
        blank_t[0:min(int_sent.size, seq_len)] = int_tags[0:min(int_sent.size, seq_len)]

        # print(int_sent[0:min(int_sent.size, seq_len)].shape)

        b_lens.append(len(s))
        b_sents.append(blank_s)
        b_tags.append(blank_t)

    lens = np.array(b_lens, dtype=np.int32)
    sentences = np.stack(b_sents)
    pos_tags = np.stack(b_tags)

    batch = []
    for i in range(n_sents // batch_size):
        batch.append((sentences[i * batch_size: i * batch_size + batch_size, :],
                      pos_tags[i * batch_size: i * batch_size + batch_size, :],
                      lens[i * batch_size: i * batch_size + batch_size]))

    return batch


data_p = sys.argv[1]
test_p = sys.argv[2]
model_loc = sys.argv[3]
target_task = sys.argv[4]
epochs = int(sys.argv[5])
gpu_mem = float(sys.argv[6])
max_len = 40

s_sents, s_tags, tagset, tagmap = read_data(data_p)
t_sents, t_tags, _, _ = read_data(test_p)

if target_task == 'pos':
    print("Choosing POS")
    target = s_tags
    t_map = tagmap
    test = t_tags
else:
    print("Choosing Chunk")
    target = s_chunks
    t_map = chunkmap
    test = t_chunks

i_t_map = dict()
for t, i in t_map.items():
    i_t_map[i] = t

print("Loading vectors")
w2v_model, w2i, init_vectors = load_model(model_loc)

print("Reading data")
batches = create_batches(128, max_len, s_sents, target, w2i, t_map)
test_batch = create_batches(len(t_sents), max_len, t_sents, test, w2i, t_map)[0]

hold_out = test_batch

print("Assembling model")
terminals = assemble_model(init_vectors, max_len, len(t_map), out_tag=tagmap['O'])







print("Starting training")
from tensorflow import GPUOptions
gpu_options = GPUOptions(per_process_gpu_memory_fraction=gpu_mem)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    summary_writer = tf.summary.FileWriter("model/", graph=sess.graph)
    for e in range(epochs):
        for ind, batch in enumerate(batches):

            sentences, pos_tags, lens = batch

            sess.run(terminals['train'],{
                terminals['words']: sentences,
                terminals['labels']: pos_tags,
                terminals['lengths']: lens
            })

            if ind % 10 == 0:

                sentences, pos_tags, lens = hold_out

                loss_val, acc_val, am = sess.run([terminals['loss'], terminals['accuracy'], terminals['argmax']], {
                    terminals['words']: sentences,
                    terminals['labels']: pos_tags,
                    terminals['lengths']: lens
                })

        # print(t_sents[0])
        # print(test[0])
        # print([i_t_map[i] for i in am[0, :lens[0]]])

        print("Epoch %d, loss %.4f, acc %.4f" % (e, loss_val, acc_val))

# lens = map(lambda x: len(x), sents)
# for w, c in Counter(lens).most_common():
#     print(w,c)
print(len(tagset))
print(len(s_sents))
