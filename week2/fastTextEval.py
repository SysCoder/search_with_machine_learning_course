import fasttext
import csv

model = fasttext.load_model("/workspace/datasets/fasttext/title_model.bin")

file = open('/workspace/datasets/fasttext/top_words.txt', 'r')

with open('/workspace/datasets/fasttext/synonyms.csv', 'w', newline='') as csvfile:
    linewriter = csv.writer(csvfile)

    for line in file.readlines():
        line = line.strip()
        print(line)
        nearest_neighbors = model.get_nearest_neighbors(line)

        filtered_nearest_neighbors = filter(lambda tuple: tuple[0] >= 0.8, nearest_neighbors)
        word_only_nearest_neighbors = [x[1] for x in filtered_nearest_neighbors]
        print(word_only_nearest_neighbors)
        row = []
        row.append(line)
        row.extend(word_only_nearest_neighbors)
        linewriter.writerow(row)


# # Train model
# model = fasttext.train_supervised(input="cooking.train")

# # Test single prediction
# model.predict("easy recipe for sourdough bread ?")

# # Evaluate on test data
# model.test("cooking.test")

# # Retrain with 25 epochs, bigrams, and learning rate of 1.0 and evaluate again
# model = fasttext.train_supervised(input="cooking.train", lr=1.0, epoch=25, wordNgrams=2)
# model.test("cooking.test")
