from langchain.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load JSON data from a file
file_path = ""  # Replace with your actual file path
loader = JSONLoader(file_path=file_path, jq_schema=".", text_content=False)
docs = loader.load()

# Convert JSON documents to text format (Assuming text is stored under a key like "content")
text_data = [doc.page_content for doc in docs]

# Initialize a text splitter (Customizable for different slicing needs)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,  # Customize based on your requirement
    chunk_overlap=20
)

# Slice the text into smaller chunks
sliced_texts = splitter.split_text("\n".join(text_data))

# Output the sliced texts
for i, chunk in enumerate(sliced_texts):
    print(f"Chunk {i+1}:\n{chunk}\n{'-'*40}")
