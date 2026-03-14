import gensim.downloader

model = gensim.downloader.load("glove-wiki-gigaword-50")
# print(model.most_similar("tower"))
print(model.similarity("cat", "cat"))
