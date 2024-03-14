def counter(filename='song.txt'):
    word_counter = {}
    with open('song.txt', 'r') as file:
        for line in file:
            words = line.split()
            for word in words:
                word = word.lower()
                word_counter[word] = word_counter.get(word, 0) + 1
    return word_counter

def main():
    word_counter = counter("song.txt")

    single_occurrence_words = [word for word, count in word_counter.items() if count == 1]

    print("Broj riječi koje se pojavljuju samo jednom:", len(single_occurrence_words))
    print("Riječi koje se pojavljuju samo jednom:", single_occurrence_words)

main()