from langchain_community.document_loaders import PyPDFLoader

docs = PyPDFLoader("syllabus.pdf").load()

print(docs[0].page_content)
print(docs[1].metadata)

# ===============================================================================
#                        LANGCHAIN PDF LOADERS REFERENCE
# ===============================================================================
# 
# Use Case                        | Recommended Loader
# --------------------------------|----------------------------------------------
# Simple, clean PDFs              | PyPDFLoader
# PDFs with tables/columns        | PDFPlumberLoader
# Scanned/image PDFs              | UnstructuredPDFLoader or AmazonTextractPDFLoader
# Need layout and image data      | PyMuPDFLoader
# Want best structure extraction  | UnstructuredPDFLoader
# ===============================================================================