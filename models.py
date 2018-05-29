import dynet as dy
import numpy as np

def run_lstm(initial_state, input_vecs):
    s = initial_state
    out_vectors = []
    for vector in input_vecs:
        s = s.add_input(vector)
        out_vector = s.output()
        out_vectors.append(out_vector)
    return out_vectors, list(s.h())


class Saveable(object):
    def __init__(self, parameter_collection, **args):
        self.args = dict(locals())
        self.args.pop("self", None)
        self.args.pop("__class__", None)
        self.args.pop("parameter_collection", None)

        pc = parameter_collection.add_subcollection()

        # add params here
        self.pc = pc


    def param_collection(self):
        return self.pc


    @classmethod
    def from_spec(cls, spec, parameter_collection):
        return cls(parameter_collection, **spec)


class VocabEmbeddings(dy.Saveable):
    def __init__(self, model, vocab, embedding_dim, BOS='<S>', EOS='</S>', UNK='<UNK>', update_embeddings=True):
        # vocab: set of strings

        self.BOS = BOS
        self.EOS = EOS
        self.UNK = UNK

        self.special_tokens = [BOS, EOS, UNK]

        vocab_set = set(vocab)
        for t in self.special_tokens:
            if t in vocab_set:
                vocab_set.remove(t)

        self.int_to_word = self.special_tokens + sorted(list(vocab_set))
        self.word_to_int = {w: i for i, w in enumerate(self.int_to_word)}

        self.BOS_INDEX = self.word_to_int[self.BOS]
        self.EOS_INDEX = self.word_to_int[self.EOS]
        self.UNK_INDEX = self.word_to_int[self.UNK]

        self.vocab_size = len(self.int_to_word)
        self.embedding_dim = embedding_dim

        self.lookup = model.add_lookup_parameters((self.vocab_size, self.embedding_dim))

        self.update_embeddings = update_embeddings

    def get_components(self):
        return self.lookup,

    def restore_components(self, components):
        self.lookup = components[0]

    def index_word(self, word):
        if word in self.word_to_int:
            return self.word_to_int[word]
        else:
            try:
                # hack for deserializing old models
                return self.UNK_INDEX
            except:
                self.UNK_INDEX = self.word_to_int[self.UNK]
                return self.UNK_INDEX

    def embed_word_index(self, index):
        try:
            update_embeddings = self.update_embeddings
        except:
            self.update_embeddings = True
            update_embeddings = self.update_embeddings
        return dy.lookup(self.lookup, index, update=update_embeddings)

    def embed_word(self, word):
        return self.embed_word_index(self.index_word(word))

    def embed_sequence(self, sequence, eos_markers=False):
        return [self.embed_word(word) for word in
                ([self.BOS] + sequence + [self.EOS] if eos_markers else sequence)]


class OneHotVocabEmbeddings(VocabEmbeddings):
    def __init__(self, model, vocab, BOS='<S>', EOS='</S>', UNK='<UNK>', update_embeddings=False):
        # vocab: set of strings
        self.BOS = BOS
        self.EOS = EOS
        self.UNK = UNK

        self.special_tokens = [BOS, EOS, UNK]

        vocab_set = set(vocab)
        for t in self.special_tokens:
            if t in vocab_set:
                vocab_set.remove(t)

        self.int_to_word = self.special_tokens + sorted(list(vocab_set))
        self.word_to_int = {w: i for i, w in enumerate(self.int_to_word)}

        self.BOS_INDEX = self.word_to_int[self.BOS]
        self.EOS_INDEX = self.word_to_int[self.EOS]
        self.UNK_INDEX = self.word_to_int[self.UNK]

        self.vocab_size = len(self.int_to_word)
        self.embedding_dim = self.vocab_size

        self.lookup = model.add_lookup_parameters((self.vocab_size, self.embedding_dim))
        self.lookup.init_from_array(np.eye(self.vocab_size))

        self.update_embeddings = update_embeddings

    def embed_word_index(self, index):
        try:
            update_embeddings = self.update_embeddings
        except:
            self.update_embeddings = False # changed from VocabEmbeddings
            update_embeddings = self.update_embeddings
        return dy.lookup(self.lookup, index, update=update_embeddings)


class GlobalAttention(dy.Saveable):
    def __init__(self, model, initializer, attention_dim, dec_state_dim, z_dim):
        self.attention_dim = attention_dim
        self.dec_state_dim = dec_state_dim
        self.z_dim = z_dim

        self.p_W = model.parameters_from_numpy(initializer.initialize((self.attention_dim, self.dec_state_dim)))
        self.p_b = model.parameters_from_numpy(initializer.initialize(self.attention_dim))
        self.p_UV = model.parameters_from_numpy(initializer.initialize((self.attention_dim, self.z_dim)))
        self.p_v = model.parameters_from_numpy(initializer.initialize(self.attention_dim))

    def get_components(self):
        return self.p_W, self.p_UV, self.p_v, self.p_b

    def restore_components(self, components):
        self.p_W, self.p_UV, self.p_v, self.p_b = components

    def __call__(self, dec_hidden_state, xh_vecs):
        # enc_vecs: one vector for each input token
        # dec_hiden_vecs: represents current decoder hidden state, one vector for each layer
        if not xh_vecs:
            # TODO: fix this hack, possibly by padding xh_vecs with bos and eos
            return dy.vecInput(self.z_dim)
        s = dy.concatenate(list(dec_hidden_state))
        W = dy.parameter(self.p_W)
        b = dy.parameter(self.p_b)
        UV = dy.parameter(self.p_UV)
        v = dy.parameter(self.p_v)

        vT = dy.transpose(v)

        Ws = W * s
        attn_weights = [
            vT * dy.tanh(Ws + UV * xh + b)
            for xh in xh_vecs
        ]
        attn_dist = dy.softmax(dy.concatenate(attn_weights))
        return dy.concatenate_cols(xh_vecs) * attn_dist, attn_dist
