import time
from collections import defaultdict

"""
without priority queue:
1000 merges: 151.2s
tokens length: 1115394
encoded tokens length: 416705
compression ratio: 2.68X
"""

class PriorityQueue():
  def __init__(self):
    self.queue = defaultdict(int)
  
  def __str__(self):
    return ''.join([str(i) for i in self.queue])
  
  def set_queue(self, in_queue:defaultdict):
    self.queue = in_queue
  
  def update_queue(self, idx:int, ids:list):
    # check if value then 0, del
    self.queue[(idx, ids[3])] += 1
    self.queue[(ids[0], idx)] += 1
    self.queue[(ids[2], ids[3])] -= 1
    self.queue[(ids[0], ids[1])] -= 1

  def pop(self):
    try:
      maxv = [(None), 0]
      for k,v in self.queue.items():
        if v > maxv[1]:
          maxv = [k, v]
      del self.queue[maxv[0]]
      return maxv[0]

    except IndexError:
      print()

class BPE:
  def __init__(self, vocab_size:int):
    self.vocab_size = vocab_size
    self.merges = {}
    self.pair_freqs = PriorityQueue()

  def _get_stats(self, ids:list):
    counts = defaultdict(int)
    for pair in zip(ids, ids[1:]):
      counts[pair] += 1
    return counts

  def _merge(self, ids:list, pair:tuple, idx:int):
    newids = []
    i = 0
    while i < len(ids):
      if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
        newids.append(idx)

        # ---
        # check for error that it's not indexing to far forward or back
        # if i < len(ids) - 2 && i > 2
        self.pair_freqs.update_queue(idx, [ids[ia] for ia in range(i-1, i+3)])
        # ---

        i += 2
      else:
        newids.append(ids[i])
        i += 1
    return newids

  def train(self, tokens:list):
    self.num_merges = self.vocab_size - 256
    ids = list(tokens)
    self.pair_freqs.set_queue(self._get_stats(ids))

    for i in range(self.num_merges):
      # ---
      stats = self._get_stats(ids)
      pair_o = max(stats, key=stats.get)
      # ---
      pair = self.pair_freqs.pop()

      print(f"{pair == pair_o}: {pair} {stats[pair]}, {pair_o} {stats[pair_o]}")

      idx = 256 + i
      print(f"{i}: merging {pair} into a new token {idx}")
      ids = self._merge(ids, pair, idx)
      self.merges[pair] = idx
    return ids

  def tokenize(self, text:str):
    tokens = list(map(int, text.encode("utf-8")))
    new_ids = []
    i = 0
    while i < len(tokens):
      if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) in self.merges:
        new_ids.append(self.merges[(tokens[i], tokens[i+1])])
        i += 2
      else:
        new_ids.append(tokens[i])
        i += 1
    return new_ids

if __name__ == "__main__":
  with open ("data/tiny_shakespeare.txt", "r", encoding="utf-8") as f:
    text:str = f.read()
  tokens = text.encode("utf-8")
  tokens = list(map(int, tokens))

  bp = BPE(256 + 100)
  start = time.time()
  ids = bp.train(tokens)
  end = time.time()

  print(f"training took {end-start:.1f}s on {bp.num_merges} merges")
  print("---")
  print("tokens length:", len(tokens))
  print("encoded tokens length:", len(ids))
  print(f"compression ratio: {len(text.encode("utf-8")) / len(ids):.2f}X")
