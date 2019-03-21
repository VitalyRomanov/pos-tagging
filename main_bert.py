import tensorflow as tf
import sys
import numpy as np
from collections import Counter
from bert_provider import convert_examples_to_features, InputExample, built_bert, tokenizer, bert_config, \
    init_checkpoint


def assemble_model(embedding_layer, seq_len, n_tags, lr=0.001, train_embeddings=False):
    emb_dim = embedding_layer.shape[2]

    d_win = 5
    h2_size = 200

    tf_labels = tf.placeholder(shape=(None, seq_len), dtype=tf.int32, name="labels")
    tf_type = tf.placeholder(shape=(None, seq_len), dtype=tf.int32, name="labels")
    tf_lengths = tf.placeholder(shape=(None,), dtype=tf.int32, name="lengths")

    emb_flat = tf.reshape(embedding_layer, shape=(-1, emb_dim))

    local_h2 = tf.layers.dense(emb_flat,
                               h2_size,
                               activation=tf.nn.tanh,
                               name="dense_h2")

    tag_logits = tf.layers.dense(local_h2, n_tags, activation=None)
    logits = tf.reshape(tag_logits, (-1, seq_len, n_tags))

    bool_mask = tf.cast(tf_type, dtype=tf.bool)
    valid_logits_labels = tf.boolean_mask(logits, bool_mask)
    true_labels = tf.boolean_mask(tf_labels, bool_mask)
    true_labels_one_hot = tf.one_hot(true_labels, depth=n_tags)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=true_labels_one_hot, logits=valid_logits_labels))

    # with tf.variable_scope('loss') as l:
    #     log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(logits, tf_labels, tf_lengths)
    #     loss = tf.reduce_mean(-log_likelihood)

    train = tf.train.AdamOptimizer(lr).minimize(loss)

    # mask = tf.cast(tf_type, dtype=tf.bool)
    argmax = tf.math.argmax(valid_logits_labels, axis=-1)
    estimated_labels = tf.cast(argmax, tf.int32)
    accuracy = tf.contrib.metrics.accuracy(estimated_labels, true_labels)

    return {
        'labels': tf_labels,
        'type': tf_type,
        'lengths': tf_lengths,
        'loss': loss,
        'train': train,
        'accuracy': accuracy,
        'argmax': argmax
    }


def read_data(path):
    sents_w = [];
    sent_w = []
    sents_t = [];
    sent_t = []
    sents_c = [];
    sent_c = []

    tags = set()
    chunk_tags = set()

    with open(path, "r") as conll:
        for line in conll.read().strip().split("\n"):
            if line == '':
                sents_w.append(sent_w)
                sent_w = []
                sents_t.append(sent_t)
                sent_t = []
                sents_c.append(sent_c)
                sent_c = []
            else:
                try:
                    word, tag, chunk = line.split()
                except:
                    continue
                tags.add(tag)
                chunk_tags.add(chunk)
                sent_w.append(word.lower())
                sent_t.append(tag)
                sent_c.append(chunk)

    tags = list(tags);
    tags.sort()
    chunk_tags = list(chunk_tags);
    chunk_tags.sort()

    tagmap = dict(zip(tags, range(len(tags))))
    chunkmap = dict(zip(chunk_tags, range(len(chunk_tags))))
    return sents_w, sents_t, sents_c, tags, tagmap, chunk_tags, chunkmap


def create_batches(batch_size, train_sent, train_mask, train_lbls, train_type, train_lens):
    batches = train_sent.shape[0] // batch_size

    for i in range(batches):
        yield train_sent[i * batch_size: (i + 1) * batch_size, ...], \
              train_mask[i * batch_size: (i + 1) * batch_size, ...], \
              train_lbls[i * batch_size: (i + 1) * batch_size, ...], \
              train_type[i * batch_size: (i + 1) * batch_size, ...], \
              train_lens[i * batch_size: (i + 1) * batch_size, ...]

    yield train_sent[batches * batch_size:, ...], \
          train_mask[batches * batch_size:, ...], \
          train_lbls[batches * batch_size:, ...], \
          train_type[batches * batch_size:, ...], \
          train_lens[batches * batch_size:]


data_p = sys.argv[1]
test_p = sys.argv[2]
model_loc = sys.argv[3]
target_task = sys.argv[4]
epochs = int(sys.argv[5])
gpu_mem = float(sys.argv[6])
max_len = 40

s_sents, s_tags, s_chunks, tagset, tagmap, chunk_tags, chunkmap = read_data(data_p)
t_sents, t_tags, t_chunks, _, _, _, _ = read_data(test_p)

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

train_sents = len(s_sents)
all_sents = s_sents + t_sents
all_targets = target + test

print("Assembling model")
sess = tf.Session()
bert_model, layers, input_ids_tf, input_mask_tf = built_bert(
    bert_config=bert_config,
    init_checkpoint=init_checkpoint,
    seq_len=42)
terminals = assemble_model(layers[-1][:, 1:-1, :], max_len, len(t_map))
terminals['input_ids'] = input_ids_tf
terminals['input_mask'] = input_mask_tf

print("Reading data")

input_ids, input_mask, output_lbls, token_type, lens = convert_examples_to_features(all_sents, all_targets, t_map, max_len + 2,
                                                                        tokenizer)


train_sent = input_ids[:train_sents, ...]
train_mask = input_mask[:train_sents, ...]
train_lbls = output_lbls[:train_sents, ...]
train_type = token_type[:train_sents, ...]
train_lens = lens[:train_sents]

test_sent = input_ids[train_sents:, ...]
test_mask = input_mask[train_sents:, ...]
test_lbls = output_lbls[train_sents:, ...]
test_type = token_type[train_sents:, ...]
test_lens = lens[train_sents:]


# from tensorflow import GPUOptions
# gpu_options = GPUOptions(per_process_gpu_memory_fraction=gpu_mem)
# with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:


i_t_map = dict()
for t, i in t_map.items():
    i_t_map[i] = t

hold_out = (test_sent, test_mask, test_lbls, test_type, test_lens)



print("Starting training")

sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())
# summary_writer = tf.summary.FileWriter("model/", graph=sess.graph)
for e in range(epochs):
    for batch in create_batches(10, train_sent, train_mask, train_lbls, train_type, train_lens):
        sent, mask, lbl, type, lens = batch

        sess.run(terminals['train'], {
            terminals['input_ids']: sent,
            terminals['input_mask']: mask,
            terminals['labels']: lbl,
            terminals['type']: type,
            terminals['lengths']: lens
        })

        break

        # if ind % 10 == 0:

    sent, mask, lbl, type, lens = hold_out

    loss_val, acc_val, am = sess.run([terminals['loss'], terminals['accuracy'], terminals['argmax']], {
        terminals['input_ids']: sent[:1000,...],
        terminals['input_mask']: mask[:1000,...],
        terminals['labels']: lbl[:1000,...],
        terminals['type']: type[:1000,...],
        terminals['lengths']: lens[:1000]
    })

    print(acc_val)

    # print(t_sents[0])
    # print(test[0])
    # print([i_t_map[i] for i in am[0, :lens[0]]])

    print("Epoch %d, loss %.4f, acc %.4f" % (e, loss_val, acc_val))

# lens = map(lambda x: len(x), sents)
# for w, c in Counter(lens).most_common():
#     print(w,c)
print(len(tagset))
print(len(s_sents))
