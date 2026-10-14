from embedding import get_query_embedding


text = "Какие требования предъявляются к работам на высоте?"

vector = get_query_embedding(text)

print()
print("Embedding успешно создан!")
print("Размерность:", len(vector))
print("Первые 10 значений:")
print(vector[:10])
