import os
from langchain_text_splitters.markdown import MarkdownHeaderTextSplitter
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def build_rulebook_index(source_dir: str, persist_dir: str = "./wyrmspan_vector_store"):
    """
    Reads a directory of Markdown files, splits them semantically by headers and character length,
    embeds it using an open-source local model, and stores it in a FAISS vector index.
    """
    # Check if the source directory exists
    if not os.path.exists(source_dir):
        raise FileNotFoundError(f"The specified source directory does not exist: {source_dir}")

    all_chunks = []

    headers_to_split_on = [
        ("#", "Header_1"),
        ("##", "Header_2"),
        ("###", "Header_3"),
    ]

    header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    char_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    # loop through all Markdown files in the source directory
    for filename in os.listdir(source_dir):
        if filename.endswith(".md"):
            file_path = os.path.join(source_dir, filename)
            print(f"Processing file: {file_path}")

            # Read the Markdown content
            with open(file_path, "r", encoding="utf-8") as file:
                markdown_content = file.read()

            # Split the content semantically by headers
            header_splits = header_splitter.split_text(markdown_content)

            # Further split the header chunks by character length
            file_chunks = char_splitter.split_documents(header_splits)

            for chunk in file_chunks:
                chunk.metadata["source_file"] = filename  # Add source filename to metadata
                all_chunks.append(chunk)

    print(f"\nTotal chunks created from all files: {len(all_chunks)}")

    # Initialize embeddings using a local model
    print("Initializing embeddings using the local model...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create a FAISS vector store and add the chunks
    print("Creating FAISS vector store...")
    vector_store = FAISS.from_documents(all_chunks, embeddings)

    # Persist the vector store to disk
    vector_store.save_local(persist_dir)
    print(f"Vector store saved to: {persist_dir}")

if __name__ == "__main__":
    source_dir = "./lookup_references"
    persist_dir = "./wyrmspan_vector_store"  # Directory to save the FAISS index
    build_rulebook_index(source_dir, persist_dir)