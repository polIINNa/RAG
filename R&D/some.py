from llama_index.legacy.readers import SimpleDirectoryReader


docs = SimpleDirectoryReader(input_files=['/Users/21109090/Desktop/госпрограмма/программы/ПП 1570.pdf']).load_data()
for doc in docs:
    print(doc.text, '\n\n')
