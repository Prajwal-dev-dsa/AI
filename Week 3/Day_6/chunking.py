from langchain_text_splitters import CharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

with open("document.txt","r",encoding="utf-8") as file:
    text = file.read()

# Character-based chunking
splitter=CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
    separator=""
)

chunks=splitter.split_text(text)

for i, chunk in enumerate(chunks):
    print(f"\n--- Chunk {i + 1} ---")
    print(chunk)

# Paragraph-based chunking
splitter=CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
    separator="\n\n"
)

chunks=splitter.split_text(text)

for i, chunk in enumerate(chunks):
    print(f"\n--- Paragraph Chunk {i + 1} ---")
    print(chunk)


# Recursive chunking
splitter=RecursiveCharacterTextSplitter(
    chunk_size=30,
    chunk_overlap=20,
)

chunks=splitter.split_text(text)

for i, chunk in enumerate(chunks):
    print(f"\n--- Recursive Chunk {i + 1} ---")
    print(chunk)