from collections import defaultdict
import random
from tqdm import tqdm
import numpy as np # import jax.numpy as jnp
import json
import matplotlib.pyplot as plt

# TODO: use jax
class AutoChars:
  @staticmethod
  def pre_tokenize_str(s:str):
    # Ġ: space, Ċ: newline
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
    if c_word: words.append(c_word)
    return words

# TODO: types
class BPE:
  def __init__(self, corpus:str, g_vocab_size:int=500, from_file:bool=False):
    self.corpus, self.g_vocab_size = corpus, g_vocab_size
    self.from_file = from_file
    self.tokenizer = AutoChars
    if not from_file:
      self.word_freqs = self._get_word_freqs()
      self.vocab = sorted(set(c for word in self.word_freqs for c in word))
      self.splits = {w: [c for c in w] for w in self.word_freqs.keys()}
      self.merges = defaultdict(str)
      self._train()
      # call save?
    else:
      self.merges = self._load_merges()

    assert len(self.merges) == g_vocab_size

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

  def _merge_pair(self, a:str, b:str):
    for word, split in self.splits.items():
      if len(split) > 1:
        i = 0
        while i < len(split) - 1:
          if split[i] == a and split[i+1] == b:
            split[i:i+2] = [a+b]
          else: i += 1

  # TODO: wrong because it includes the original vocab as well (single chars)
  # TODO: optimize
  def _train(self):
    with tqdm(total=self.g_vocab_size, desc='training') as pbar:
      while len(self.merges) < self.g_vocab_size:
        pair_freqs = self._compute_pair_freqs()
        best_pair = max(pair_freqs, key=pair_freqs.get)
        self._merge_pair(*best_pair)
        self.merges[''.join(best_pair)] = best_pair
        pbar.update(1)

  def tokenize(self, txt:str):
    splits = [list(w) for w in self.tokenizer.pre_tokenize_str(txt)]
    # TODO: optimize, going through each pair in merges for each in split
    for token, pair in tqdm(self.merges.items(), desc='tokenizing'):
      for split in splits:
        i = 0
        while i < len(split)-1:
          if split[i:i+2] == list(pair):
            split[i:i+2] = [token]
          else: i += 1
    return [it for sl in tqdm(splits) for it in sl]

  def save_merges(self, path:str='merges.json'):
    with open(path, 'w', encoding='utf-8') as fp:
      json.dump(self.merges, fp)

  def _load_merges(self, path:str='merges.json'):
    with open(path, 'r', encoding='utf-8') as fp:
      data = json.load(fp)
    return data

class Drop_BPE(BPE):
  def __init__(self, corpus:str, g_vocab_size:int=500, drop_range:float=0.4, drop_rate:float=0.4, drop_pos:float=0.0, from_file:bool=False):
    # TODO: add param for random drop window location on distribution
    super().__init__(corpus, g_vocab_size, from_file=from_file)
    self.drop_range, self.drop_rate = drop_range, drop_rate
    if not from_file:
      self.corpus_tokenized = self.tokenize(self.corpus)
    else:
      self.corpus_tokenized = self._load_corpus_tokenized()
    self.token_frequencies = self.get_token_frequencies()

  def get_token_frequencies(self):
    b = {}
    for w in tqdm(self.corpus_tokenized):
      if w in self.merges.keys():
        b[w] = b.get(w, 0) + 1
    return sorted(b, key=b.get, reverse=True)

  def rnd_drop(self):
    N = int(len(self.merges.keys())*self.drop_range)
    k = int(N*0.4)
    assert k < N and N < len(self.merges)
    for i in range(k):
      ix = random.randint(0, N)
      # TODO: this check if provisionary for a KeyError, problem is removing same thing twice
      if self.token_frequencies[ix] in self.merges.keys():
        print('deleting:', self.token_frequencies[ix])
        del(self.merges[self.token_frequencies[ix]]) # fix this
      else:
        i -= 1
    if self.from_file:
      self.save_merges()

  def save_corpus_tokenized(self, path:str='corpus_tokenized.txt'):
    with open(path, 'w', encoding='utf-8') as fp:
      json.dump(self.corpus_tokenized, fp)

  def _load_corpus_tokenized(self, path:str='corpus_tokenized.txt'):
    with open(path, 'r', encoding='utf-8') as fp:
      data = json.load(fp)
      print(type(data), len(data))
      print(data)
    return data

if __name__ == '__main__':
  with open ('data/tiny_shakespeare.txt', 'r', encoding='utf-8') as f:
    bp = Drop_BPE(f.read(), 5000)

  exit(1)

  bp.rnd_drop()
  b = bp.get_token_frequencies()
  vals = np.log(np.array([*b.values()]))
  tokens = np.arange(0, len(b.keys()))

  plt.bar(tokens, vals)
  plt.title(f'random drop (max: {max(b.values())} total: {len(b.values())})')
  plt.xlabel('token #')
  plt.ylabel('log frequency')
  plt.savefig('rnd_drop_bpe.png')
  plt.show()
