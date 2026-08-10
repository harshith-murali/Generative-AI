from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(
    file_path="data.csv",
    encoding="utf-8",
    csv_args={
        "delimiter": ",",
        "quotechar": '"',
    },
)

docs = loader.load()

print(f"Total rows loaded: {len(docs)}")
print(docs[0].page_content)
print(docs[0].metadata)