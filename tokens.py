# adpated from: https://huggingface.co/learn/nlp-course/en/chapter6/5
from transformers import AutoTokenizer # TODO: implement this custom as well
from collections import defaultdict
from tqdm import tqdm

class Tokens:
    def __init__(self, corpus: list, g_vocab_size: int=100):
        self.corpus, self.g_vocab_size = corpus, g_vocab_size
        self.tokenizer = AutoTokenizer.from_pretrained('gpt2')

        self.word_freqs = self._get_word_freqs()
        self.vocab = sorted(set(c for word in self.word_freqs for c in word))
        self.splits = {w: [c for c in w] for w in self.word_freqs.keys()}
        self.merges = self._train()

        self.stoi = {p:i for i,p in enumerate(self.vocab)}
        self.itos = {i:p for i,p in enumerate(self.vocab)}
        self.encode = lambda s: [self.stoi[c] for c in s]
        self.decode = lambda e: ''.join([self.itos[i] for i in e])

        self.detokenize = lambda ts: ''.join(t.replace('Ġ', ' ') for t in ts)

    def _get_word_freqs(self):
        word_freqs = defaultdict(int)
        for txt in self.corpus:
            for word, _ in self.tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(txt):
                word_freqs[word] += 1
        return word_freqs

    def _compute_pair_freqs(self):
        pair_freqs = defaultdict(int)
        for word, freq in self.word_freqs.items():
            split = self.splits[word]
            if len(split) > 1:
                for i in range(len(split)-1):
                    pair_freqs[(split[i], split[i+1])] += freq
        return pair_freqs

    def _merge_pair(self, a: str, b: str):
        for word, split in self.splits.items():
            if len(split) > 1:
                i = 0
                while i < len(split) - 1:
                    if split[i] == a and split[i+1] == b:
                        split[i:i+2] = [a+b]
                    else: i += 1

    def _train(self):
        merges = defaultdict(str)
        with tqdm(total=self.g_vocab_size) as pbar:
            while len(self.vocab) < self.g_vocab_size:
                pair_freqs = self._compute_pair_freqs()
                best_pair = max(pair_freqs, key=pair_freqs.get)
                self._merge_pair(*best_pair)
                merges[best_pair] = ''.join(best_pair)
                self.vocab.append(''.join(best_pair))
                pbar.update(1)
            return merges

    # check if str or list of strs
    def tokenize(self, txt: str):
        splits = [list(w) for w, _ in self.tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(txt)]
        for pair, merge in self.merges.items():
            for split in splits:
                i = 0
                while i < len(split)-1:
                    if split[i:i+2] == list(pair):
                        split[i:i+2] = [merge]
                    else: i += 1
        return sum(splits, [])