import sys
from collections import Counter

editdistance2val = Counter()

with open("log.txt", "w") as log:
	editdistance2val = Counter()
	
	for j in range(0,6):
		log.write("Edit distance threshold: {}".format(j)+"\n")
		threshold = j
		for i in range(1,2):
		
			f = open("preds" + str(i) + ".txt", "r")
			lines = f.readlines()
			f.close()

			good, bad = 0., 0.
			avg_dis = 0.
			normalized_edit_distance = 0.
			total_length = 0.

			for k, line in enumerate(lines):
		
				#if i == 0: continue
	
				parts = line.strip().split("\t")
				normalized_edit_distance += float(parts[-1])
				edit_distance = int(parts[-2])
				latin = parts[-3]
				latin = latin.replace("<","").replace(">","")
				total_length += len(latin)
				avg_dis += edit_distance

				if edit_distance <= threshold:
					good += 1
				else:
					bad += 1
				
				if j == 0:
					editdistance2val[edit_distance] += 1
		
			acc = good / (good + bad)
			acc = "{0:.3f}".format(acc)
			avg_dis /= len(lines)
			avg_dis = "{0:.3f}".format(avg_dis)
			avg_normalized_dis =  "{0:.3f}".format(normalized_edit_distance / len(lines))
		
			log.write("run number {}; acc: {}; average edit distance: {}; average normalized edit distnace: {}".format(i, acc, avg_dis, avg_normalized_dis) + "\n")
		log.write("------------------------------------\n")
	props = [float("{0:.3f}".format(x/(5. * len(lines)))) for x in editdistance2val.values()]
	log.write("error distribution: {}".format(props) + "\n")

