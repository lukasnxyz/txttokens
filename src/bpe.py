from collections import defaultdict

class AutoChars:
    @staticmethod
    def pre_tokenize_str(s: str):
        words = []
        c_word = ''
        for l in s:
            if not l.isalpha():
                if l == ' ':
                    if c_word == '': c_word += 'Ġ'
                    else:
                        words.append(c_word)
                        c_word = 'Ġ'
                elif l == '\n':
                    if c_word != '': words.append(c_word)
                    words.append('Ċ')
                    c_word = ''
                else: c_word += l
            else: c_word += l
        return words

class Tokens:
    def __init__(self, corpus: str, g_vocab_size: int=100):
        self.corpus, self.g_vocab_size = corpus, g_vocab_size
        self.tokenizer = AutoChars

        self.word_freqs = self._get_word_freqs()
        self.vocab = sorted(set(c for word in self.word_freqs for c in word))
        self.splits = {w: [c for c in w] for w in self.word_freqs.keys()}
        self.merges = self._train()

        self.stoi = {p:i for i,p in enumerate(self.vocab)}
        self.itos = {i:p for i,p in enumerate(self.vocab)}
        self.encode = lambda s: [self.stoi[c] for c in s]
        self.decode = lambda e: ''.join([self.itos[i] for i in e])

        self.detokenize = lambda ts: ''.join(t.translate(str.maketrans({'Ġ': ' ', 'Ċ': '\n'})) for t in ts)

    def _get_word_freqs(self):
        word_freqs = defaultdict(int)
        for word in self.tokenizer.pre_tokenize_str(self.corpus):
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
        while len(self.vocab) < self.g_vocab_size:
            print(f'{len(self.vocab)}/{self.g_vocab_size}', end='\r', flush=True)
            pair_freqs = self._compute_pair_freqs()
            best_pair = max(pair_freqs, key=pair_freqs.get)
            self._merge_pair(*best_pair)
            merges[best_pair] = ''.join(best_pair)
            self.vocab.append(''.join(best_pair))
        return merges

    # TODO: some sort of progress
    def tokenize(self, txt: str):
        splits = [list(w) for w in self.tokenizer.pre_tokenize_str(txt)]
        for pair, merge in self.merges.items():
            for split in splits:
                i = 0
                while i < len(split)-1:
                    if split[i:i+2] == list(pair):
                        split[i:i+2] = [merge]
                    else: i += 1
        return sum(splits, [])
