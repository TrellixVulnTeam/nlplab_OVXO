from __future__ import absolute_import
from __future__ import division

import re
from tensorflow.python.platform import gfile
import tensorflow as tf
import os
import numpy as np
import random

FLAGS = tf.app.flags.FLAGS

# Special vocabulary symbols - we always put them at the start.
_PAD = b"_PAD"
_SOS = b"_SOS"
_EOS = b"_EOS"
_UNK = b"_UNK"
_START_VOCAB = [_PAD, _SOS, _EOS, _UNK]

PAD_ID = 0
SOS_ID = 1
EOS_ID = 2
UNK_ID = 3

# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")
_DIGIT_RE = re.compile(br"\d")


def basic_tokenizer(sentence):
    """Very basic tokenizer: split the sentence into a list of tokens."""
    # used by NLCEnv
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(re.split(_WORD_SPLIT, space_separated_fragment))
    return [w for w in words if w]


def char_tokenizer(sentence):
    # used by NLCEnv
    return list(sentence.strip())


def tokenize(string):
    # used by util code
    return [int(s) for s in string.split()]


def pair_iter(fnamex, fnamey, batch_size, num_layers):
    fdx, fdy = open(fnamex), open(fnamey)
    batches = []

    while True:
        if len(batches) == 0:
            refill(batches, fdx, fdy, batch_size)
        if len(batches) == 0:
            break

        x_tokens, y_tokens = batches.pop(0)
        y_tokens = add_sos_eos(y_tokens)
        x_padded, y_padded = padded(x_tokens, num_layers), padded(y_tokens, 1)

        source_tokens = np.array(x_padded).T
        source_mask = (source_tokens != PAD_ID).astype(np.int32)
        target_tokens = np.array(y_padded).T
        target_mask = (target_tokens != PAD_ID).astype(np.int32)

        yield (source_tokens, source_mask, target_tokens, target_mask)

    return


def refill(batches, fdx, fdy, batch_size):
    line_pairs = []
    linex, liney = fdx.readline(), fdy.readline()

    while linex and liney:
        x_tokens, y_tokens = tokenize(linex), tokenize(liney)

        if len(x_tokens) < FLAGS.max_seq_len and len(y_tokens) < FLAGS.max_seq_len:
            line_pairs.append((x_tokens, y_tokens))
        if len(line_pairs) == batch_size * 16:
            break
        linex, liney = fdx.readline(), fdy.readline()

    line_pairs = sorted(line_pairs, key=lambda e: len(e[0]))

    for batch_start in xrange(0, len(line_pairs), batch_size):
        x_batch, y_batch = zip(*line_pairs[batch_start:batch_start + batch_size])
        #    if len(x_batch) < batch_size:
        #      break
        batches.append((x_batch, y_batch))

    random.shuffle(batches)
    return


def add_sos_eos(tokens):
    return map(lambda token_list: [SOS_ID] + token_list + [EOS_ID], tokens)


def padded(tokens, depth):
    maxlen = max(map(lambda x: len(x), tokens))
    align = pow(2, depth - 1)
    padlen = maxlen + (align - maxlen) % align
    return map(lambda token_list: token_list + [PAD_ID] * (padlen - len(token_list)), tokens)


def preprocess_nlc_data(data_dir, max_vocabulary_size, custom_appendix="", tokenizer=char_tokenizer):
    """
    Parameters
    data_dir: with ending slash
    max_vocabulary_size: normally 10k
    custom_appendix: like ".30.", should be seperated by period

    We preprocess our corpus in a similar manner like in NLC paper
    Notice that this method generates indices files
    and does NOT generate batched data but store index files
    """

    train_path = data_dir + "train" + custom_appendix
    dev_path = data_dir + "valid" + custom_appendix

    # Create vocabularies of the appropriate sizes.
    vocab_path = os.path.join(data_dir, "vocab.dat")
    create_vocabulary(vocab_path, [train_path + ".y.txt", train_path + ".x.txt"],
                             max_vocabulary_size, tokenizer)

    # Create token ids for the training data.
    y_train_ids_path = train_path + ".ids.y"
    x_train_ids_path = train_path + ".ids.x"
    data_to_token_ids(train_path + ".y.txt", y_train_ids_path, vocab_path, tokenizer)
    data_to_token_ids(train_path + ".x.txt", x_train_ids_path, vocab_path, tokenizer)

    # Create token ids for the development data.
    y_dev_ids_path = dev_path + ".ids.y"
    x_dev_ids_path = dev_path + ".ids.x"
    data_to_token_ids(dev_path + ".y.txt", y_dev_ids_path, vocab_path, tokenizer)
    data_to_token_ids(dev_path + ".x.txt", x_dev_ids_path, vocab_path, tokenizer)

    return (x_train_ids_path, y_train_ids_path,
            x_dev_ids_path, y_dev_ids_path, vocab_path)


def create_vocabulary(vocabulary_path, data_paths, max_vocabulary_size,
                      tokenizer=None, normalize_digits=False):
    if not gfile.Exists(vocabulary_path):
        print("Creating vocabulary %s from data %s" % (vocabulary_path, str(data_paths)))
        vocab = {}
        for path in data_paths:
            with gfile.GFile(path, mode="rb") as f:
                counter = 0
                for line in f:
                    counter += 1
                    if counter % 100000 == 0:
                        print("  processing line %d" % counter)
                    tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
                    for w in tokens:
                        word = re.sub(_DIGIT_RE, b"0", w) if normalize_digits else w
                        if word in vocab:
                            vocab[word] += 1
                        else:
                            vocab[word] = 1
        vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
        print("Vocabulary size: %d" % len(vocab_list))
        if len(vocab_list) > max_vocabulary_size:
            vocab_list = vocab_list[:max_vocabulary_size]
        with gfile.GFile(vocabulary_path, mode="wb") as vocab_file:
            for w in vocab_list:
                vocab_file.write(w + b"\n")


def initialize_vocabulary(vocabulary_path):
    if gfile.Exists(vocabulary_path):
        rev_vocab = []
        with gfile.GFile(vocabulary_path, mode="rb") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip('\n') for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def sentence_to_token_ids(sentence, vocabulary,
                          tokenizer=None, normalize_digits=False):
    if tokenizer:
        words = tokenizer(sentence)
    else:
        words = basic_tokenizer(sentence)
    if not normalize_digits:
        return [vocabulary.get(w, UNK_ID) for w in words]
        # Normalize digits by 0 before looking words up in the vocabulary.
    return [vocabulary.get(re.sub(_DIGIT_RE, b"0", w), UNK_ID) for w in words]


def data_to_token_ids(data_path, target_path, vocabulary_path,
                      tokenizer=None, normalize_digits=True):
    if not gfile.Exists(target_path):
        print("Tokenizing data in %s" % data_path)
        vocab, _ = initialize_vocabulary(vocabulary_path)
        with gfile.GFile(data_path, mode="rb") as data_file:
            with gfile.GFile(target_path, mode="w") as tokens_file:
                counter = 0
                for line in data_file:
                    counter += 1
                    if counter % 100000 == 0:
                        print("  tokenizing line %d" % counter)
                    token_ids = sentence_to_token_ids(line, vocab, tokenizer,
                                                             normalize_digits)
                    tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")

if __name__ == '__main__':
    # ======= Test on Vocab/ind Creation ========
    # nlc char-level vocab: 122
    # ptb: Vocabulary size: 52
    x_train, y_train, x_dev, y_dev, vocab_path \
        = preprocess_nlc_data(data_dir="/Users/Aimingnie/Documents/School/Stanford/AA228/nlplab/ptb_data/", max_vocabulary_size=40000)
    print x_train
    print y_train
    print x_dev
    print y_dev
    print vocab_path
