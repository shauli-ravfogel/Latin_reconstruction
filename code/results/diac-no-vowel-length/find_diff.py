with open("preds1.txt", "r") as f:

	lines = f.readlines()

with open("preds-orto-no-diac-embs-cyclic.txt", "r") as f:

	embs_lines = f.readlines()
	

words_0 = set()
w2pred = {}
words_0_embs = set()

for line in lines:

	line = line.strip().split("\t")
	w, ed = line[-3], line[-2]
	rec = line[-4]
	w2pred[w] = rec
	if ed == "0":
	
		words_0.add(w)
		
for line in embs_lines:

	line = line.strip().split("\t")
	w, ed = line[-3], line[-2]

	if int(ed) == 0:
	
		words_0_embs.add(w)


for w in words_0_embs:

	if w not in words_0:
	
		print(w + "\t" + w2pred[w])
