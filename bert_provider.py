import bert
import codecs
import collections
from bert import tokenization
from bert import modeling
import tensorflow as tf
import numpy as np
from bert.extract_features import InputExample, convert_examples_to_features, input_fn_builder, _truncate_seq_pair

max_seq_len = 40
pred_batch_size = 32
init_checkpoint = "multi_cased_L-12_H-768_A-12/bert_model.ckpt"
config_path = "multi_cased_L-12_H-768_A-12/bert_config.json"
vocab_path = "multi_cased_L-12_H-768_A-12/vocab.txt"
layer_indexes = [-1,-2,-3,-4]
tokenizer = tokenization.FullTokenizer(
      vocab_file=vocab_path, do_lower_case=False)


bert_config = modeling.BertConfig.from_json_file(config_path)
tokenizer = tokenization.FullTokenizer(
      vocab_file=vocab_path, do_lower_case=False)
is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
run_config = tf.contrib.tpu.RunConfig(
    master=None,
    tpu_config=tf.contrib.tpu.TPUConfig(
    num_shards=1,
    per_host_input_for_training=is_per_host))
sess = tf.Session()


def convert_examples_to_features(examples, all_tags, tagmap, seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    features = []

    input_ids = np.zeros((len(examples), seq_length), dtype=np.int32)
    input_mask = np.zeros((len(examples), seq_length), dtype=np.int32)
    output_lbls = np.zeros((len(examples), seq_length-2), dtype=np.int32)

    lengths = []

    for (ex_index, example) in enumerate(examples):
        tags = all_tags[ex_index]

        tokens = ["[CLS]"] + example

        bert_tokens = []
        bert_labels = []

        for ind, token in enumerate(tokens):
            shards = tokenizer.tokenize(token) if token not in {"[CLS]", "[SEP]"} else [token]
            if len(bert_tokens) + len(shards) >= seq_length: break
            tkns = tokenizer.convert_tokens_to_ids(shards)
            bert_tokens.extend(tkns)
            # skip special tokens
            if token in {"[CLS]", "[SEP]"}: continue
            # extend label to token segments
            if len(tkns) == 1:
                bert_labels.extend([tags[ind-1]])
            else:
                if len(tags[ind-1][:2]) > 2 and tags[ind-1][:2] == 'B-':
                    bert_labels.extend([tags[ind-1]] + ['I-'+tags[ind-1][2:]] * (len(tkns)-1))
                else:
                    bert_labels.extend([tags[ind-1]] * len(tkns))
        tkns = tokenizer.convert_tokens_to_ids(["[SEP]"])
        bert_tokens.extend(tkns)

        token_ids = np.array(bert_tokens)
        lbls = np.array([tagmap.get(l, 0) for l in bert_labels])
        input_ids[ex_index, :token_ids.size] = token_ids
        input_mask[ex_index, :token_ids.size] = 1
        output_lbls[ex_index, :lbls.size] = lbls
        lengths.append(lbls.size)

    return input_ids, input_mask, output_lbls, np.array(lengths, dtype=np.int32)


# ids, mask = convert_examples_to_features(
#       examples=examples, seq_length=max_seq_len, tokenizer=tokenizer)


def built_bert(bert_config,
           init_checkpoint,
           use_one_hot_embeddings=False,
           seq_len=42):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    input_ids = tf.placeholder(shape=(None, seq_len), dtype=tf.int32, name="input_ids")
    input_mask = tf.placeholder(shape=(None, seq_len), dtype=tf.int32, name="input_mask")
    input_type_ids = None

    model = modeling.BertModel(
        config=bert_config,
        is_training=False,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=input_type_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    tvars = tf.trainable_variables()
    scaffold_fn = None
    (assignment_map,
     initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(
         tvars, init_checkpoint)

    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    # tf.logging.info("**** Trainable Variables ****")
    # for var in tvars:
    #   init_string = ""
    #   if var.name in initialized_variable_names:
    #     init_string = ", *INIT_FROM_CKPT*"
    #   tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
    #                   init_string)

    all_layers = model.get_all_encoder_layers()

    return model, all_layers, input_ids, input_mask


# bert_model, layers, input_ids_tf, input_mask_tf = built_bert(
#       bert_config=bert_config,
#       init_checkpoint=init_checkpoint)
# sess.run(tf.global_variables_initializer())
# sess.run(tf.local_variables_initializer())