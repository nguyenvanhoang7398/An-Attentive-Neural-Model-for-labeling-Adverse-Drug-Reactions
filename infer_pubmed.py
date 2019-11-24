import utils
from constants import *
from nltk.tokenize import TweetTokenizer
import json
from keras.preprocessing.sequence import pad_sequences
from run_pubmed import build_model
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
import numpy as np


def preprocess_input_doc(docs, tokenizer, word2idx, char2idx, max_len, max_len_char_word):

    # word-level preprocessing
    tokenized_docs = []
    doc_word_idxs = []
    for doc in docs:
        tokens = tokenizer.tokenize(doc)
        lowered_tokens = [t.lower() for t in tokens]
        tokenized_docs.append(lowered_tokens)
        token_idxs = [word2idx[word] if word in word2idx else 1 for word in lowered_tokens]     # 'UNK' idx is 1
        doc_word_idxs.append(token_idxs)
    padded_doc_word_idxs = pad_sequences(doc_word_idxs, maxlen=max_len)

    # char-level preprocessing from original codes, leave untouched
    char_per_word = []
    char_word = []
    char_senc = []

    for sentence in tokenized_docs:
        for word in sentence:
            for c in word.lower():
                char_per_word.append(c)
            if len(char_per_word) > 37:
                char_per_word = char_per_word[:37]
            char_word.append(char_per_word)
            char_per_word = []
        char_senc.append(char_word)
        char_word = []

    char_word_lex = []
    char_lex = []
    char_word = []
    for senc in char_senc:
        for word in senc:
            for charac in word:
                char_word_lex.append([char2idx[charac]])

            char_word.append(char_word_lex)
            char_word_lex = []

        char_lex.append(char_word)
        char_word = []

    char_per_word = []
    char_per_senc = []
    char_senc = []
    for s in char_lex:
        for w in s:
            for c in w:
                for e in c:
                    char_per_word.append(e)
            char_per_senc.append(char_per_word)
            char_per_word = []
        char_senc.append(char_per_senc)
        char_per_senc = []

    pad_char_all = []
    for senc in char_senc:
        while len(senc) < max_len:
            senc.insert(0, [])
        pad_char_all.append(pad_sequences(senc, maxlen=max_len_char_word))

    pad_char_all = np.array(pad_char_all)

    return padded_doc_word_idxs, pad_char_all, tokenized_docs


if __name__ == "__main__":
    input_docs = utils.load_text_as_list(INPUT_DOCS_PATH)
    input_tokenizer = TweetTokenizer()
    with open(META_PATH, "r") as f:
        meta = json.load(f)
    input_word2idx = meta["word2idx"]
    input_max_len = meta["max_len"]
    input_max_char_len = meta["max_char_len"]
    input_char2idx = meta["char2idx"]
    input_vocab_size = meta["vocsize"]
    input_char_vocab_size = meta["charsize"]
    input_embed_dim = meta["embed_dim"]
    input_char_embed_dim = meta["char_embed_dim"]
    input_num_classes = meta["num_classes"]
    input_idx2label = meta["idx2label"]
    input_idx2label = {int(k): v for k, v in input_idx2label.items()}

    # bug_input_word_idxs, bug_input_char_idxs, bug_tokenized_docs = preprocess_input_doc(input_docs, input_tokenizer,
    #                                                                                     input_word2idx, input_char2idx,
    #                                                                                     input_max_len, input_max_char_len)
    input_word_idxs, input_char_idxs, tokenized_docs = preprocess_input_doc(input_docs, input_tokenizer,
                                                                                        input_word2idx, input_char2idx,
                                                                                        input_max_len,
                                                                                        input_max_char_len)
    # input_word_idxs = utils.load_from_pickle("test_lex.pickle")
    # input_char_idxs = utils.load_from_pickle("pad_test_lex.pickle")
    # tokenized_docs = utils.load_from_pickle("test_toks.pickle")

    model_checkpoint = "output/extractor/model-0010.ckpt"
    input_idx2word = dict((k, v) for v, k in input_word2idx.items())

    input_idx2word[0] = 'PAD'
    input_idx2word[1] = 'UNK'

    print('Loading word embeddings...')
    _ = glove2word2vec(glove_300d_path, glove_300d_tmp_path)
    w2v = KeyedVectors.load_word2vec_format(glove_300d_tmp_path, binary=False, unicode_errors='ignore')
    print('word embeddings loading done!')

    model = build_model(input_max_len, input_max_char_len, input_idx2word, w2v,
                        input_vocab_size, input_char_vocab_size, input_embed_dim, input_char_embed_dim,
                        input_num_classes)

    model.load_weights(model_checkpoint)
    pred_probs = model.predict([input_word_idxs, input_char_idxs], verbose=0)
    pred = np.argmax(pred_probs, axis=2)
    infer_output_path = "prediction.out"
    with open(infer_output_path, 'w') as fout:
        for i in range(len(tokenized_docs)):
            bos = 'BOS\tO\tO\n'
            fout.write(bos)

            sentlen = len(tokenized_docs[i])
            startind = input_max_len - sentlen

            preds = [input_idx2label[j] for j in pred[i][startind:]]
            for (w, p) in zip(tokenized_docs[i], preds):
                line = '\t'.join([w, p]) + '\n'
                fout.write(line)

            eos = 'EOS\tO\tO\n'
            fout.write(eos)
