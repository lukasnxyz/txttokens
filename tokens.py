# adpated from: https://huggingface.co/learn/nlp-course/en/chapter6/5
from transformers import AutoTokenizer
from collections import defaultdict

class Tokens:
    def __init__(self, corpus: list):
        self.corpus = corpus
        self.tokenizer = AutoTokenizer.from_pretrained('gpt2')
        self.word_freqs = None
        self.stoi = lambda s: {p:i for i,p in enumerate(s)}
        self.itos = lambda s: {i:p for i,p in enumerate(s)}
    
    def _get_word_freqs(self):
        word_freqs = defaultdict(int)
        for txt in self.corpus:
            wrds_w_offsets = self.tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(txt)
            nw_wrds = [word for word, offset in wrds_w_offsets]
            for word in nw_wrds: word_freqs[word] += 1
        self.word_freqs = word_freqs

    def get_vocab(self):
        self._get_word_freqs()
        vocab = []
        for word in self.word_freqs.keys():
            for l in word:
                if l not in vocab: vocab.append(l)
        return sorted(vocab)

    def _compute_pair_freqs(self, splits: dict):
        # assert word freqs not NOne
        pair_freqs = defaultdict(int)
        for word, freq in self.word_freqs.items():
            split = splits[word]
            if len(split) == 1: continue
            for i in range(len(split)-1):
                pair = (split[i], split[i+1])
                pair_freqs[pair] += freq
        return pair_freqs

    def _merge_pair(self, a: str, b: str, splits: dict):
        # assert word freqs not NOne
        for word in self.word_freqs:
            split = splits[word]
            if len(split) == 1: continue
            i = 0
            while i < len(split)-1:
                if split[i] == a and split[i+1] == b:
                    split = split[:i] + [a+b] + split[i+2:]
                else: i += 1
            splits[word] = split
        return splits

    def train(self, splits: dict, vocab: list, g_vocab_size: int):
        merges = defaultdict(str)
        while len(vocab) < g_vocab_size:
            print(f'{len(vocab)}/{g_vocab_size}', end='\r', flush=True)
            pair_freqs = self._compute_pair_freqs(splits)
            best_pair = ''
            max_freq = None
            for pair, freq in pair_freqs.items():
                if max_freq is None or max_freq < freq:
                    best_pair = pair
                    max_freq = freq
            splits = self._merge_pair(*best_pair, splits)
            merges[best_pair] = best_pair[0] + best_pair[1]
            vocab.append(best_pair[0] + best_pair[1])
        return splits, vocab, merges

    def tokenize(self, txt: str, merges: defaultdict):
        ptres = self.tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(txt)
        pttxt = [word for word, offset in ptres]
        splits = [[l for l in word] for word in pttxt]
        for pair, merge in merges.items():
            for idx, split in enumerate(splits):
                i = 0
                while i < len(split)-1:
                    if split[i] == pair[0] and split[i+1] == pair[1]:
                        split = split[:i] + [merge] + split[i+2:]
                    else: i += 1
                splits[idx] = split
        return sum(splits, [])

    @staticmethod
    def detokenize(tkns: list):
        return ''.join(t.replace('Ġ', ' ') for t in tkns)

    @staticmethod
    def encode(s: str, stoi: dict):
        return [stoi[c] for c in s]

    @staticmethod
    def decode(e, itos: dict):
        return ''.join([itos[i] for i in e])


    
if __name__ == '__main__':
    with open ('data/truths.txt', 'r', encoding='utf-8') as f: 
        corpus = f.read().split('\n')
        tks = Tokens(corpus)

    vocab = tks.get_vocab()
    splits = {word: [c for c in word] for word in tks.word_freqs.keys()}
    splits, vocab, merges = tks.train(splits, vocab, 1000)

    tokenized = [tks.tokenize(i, merges) for i in corpus]
    encoded = [tks.encode(i, tks.stoi(vocab)) for i in tokenized]

    print(tokenized[:10])
    print(encoded[:10])
    #print(tks.detokenize(tokenized)) # doesn't work because is a list