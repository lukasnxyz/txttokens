from tokens import Tokens

if __name__ == '__main__':
    # https://gist.github.com/lukasnxyz/d7b29398dd3b3d1c1dcb14f1f19e3744
    # truths.txt has ~4861 unique words
    with open ('data/tiny_shakespeare.txt', 'r', encoding='utf-8') as f: 
        corpus = f.read()
        tks = Tokens(corpus, 1000) # vocab, splits, merges are automatically processed

    # example on single string
    test_str = 'the earth is round and so is the sun\nbut a cube is not.'
    tokenized = tks.tokenize(test_str)
    encoded = tks.encode(tokenized)
    print(tokenized)
    print(encoded)
    print(tks.decode(encoded))
    print(tks.detokenize(tks.decode(encoded)))

    # example on whole data set
    #tokenized = [tks.tokenize(s) for s in tks.corpus]
    #encoded = [tks.encode(t) for t in tokenized]
    #print(tokenized[:5])
    #print(encoded[:5])
