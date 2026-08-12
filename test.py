from config import make_doc_id

with open("data/uploads/sample.pdf", "rb") as file:
    print(make_doc_id(file.read()))

with open("/home/daoud/Desktop/sample.pdf", "rb") as file:
    print(make_doc_id(file.read()))