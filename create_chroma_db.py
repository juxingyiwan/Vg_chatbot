# 首先导入所需第三方库
from langchain.document_loaders import UnstructuredFileLoader
from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.document_loaders import UnstructuredWordDocumentLoader
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from tqdm import tqdm

import os
import shutil

import chardet
import os

def convert_to_utf8(file_path):
    with open(file_path, 'rb') as file:
        raw_data = file.read()
        result = chardet.detect(raw_data)
        encoding = result['encoding']
    
    if encoding is not None and encoding.lower() != 'utf-8':
        with open(file_path, 'r', encoding=encoding, errors='ignore') as file:
            content = file.read()

        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(content)
        print(f"Converted {file_path} to UTF-8.")
    else:
        print(f"{file_path} is already in UTF-8.")


## pdf移动至根目录
# def copy_pdfs_to_root(source_directory):
#     for root, dirs, files in os.walk(source_directory):
#         for file in files:
#             if file.endswith(".pdf"):
#                 source_path = os.path.join(root, file)
#                 destination_path = os.path.join(source_directory, file)

#                 # 避免重名，如果根目录已存在相同名称的文件，可以添加适当的后缀或其他处理
#                 if not os.path.exists(destination_path):
#                     shutil.copy(source_path, destination_path)
#                     print(f"Copied {file} to {source_directory}")



# 获取文件路径函数
def get_files(dir_path):
    # args：dir_path，目标文件夹路径
    file_list = []
    for filepath, dirnames, filenames in os.walk(dir_path):
        # os.walk 函数将递归遍历指定文件夹
        print(filenames)
        for filename in filenames:
            # 通过后缀名判断文件类型是否满足要求
            if filename.endswith(".md"):
                # 如果满足要求，将其绝对路径加入到结果列表
                file_list.append(os.path.join(filepath, filename))
            elif filename.endswith(".txt"):
                file_list.append(os.path.join(filepath, filename))
            elif filename.endswith(".doc") or filename.endswith(".docx"):
                file_list.append(os.path.join(filepath, filename))
            elif filename.endswith(".pdf"):
                file_list.append(os.path.join(filepath, filename))
    return file_list

# 加载文件函数
def get_text(dir_path):
    # args：dir_path，目标文件夹路径
    # 首先调用上文定义的函数得到目标文件路径列表
    file_lst = get_files(dir_path)
    print(file_lst)
    # docs 存放加载之后的纯文本对象
    docs = []
    # 遍历所有目标文件
    for one_file in tqdm(file_lst):
        file_type = one_file.split('.')[-1]
        if file_type == 'md':
            convert_to_utf8(one_file)
            loader = UnstructuredMarkdownLoader(one_file)
        elif file_type == 'txt':
            convert_to_utf8(one_file)
            loader = UnstructuredFileLoader(one_file)
        elif file_type == 'doc' or file_type == 'docx':
            loader = UnstructuredWordDocumentLoader(one_file)
            print(loader.load())
        elif file_type == 'pdf':
            loader = UnstructuredPDFLoader(one_file)
        else:
            # 如果是不符合条件的文件，直接跳过
            continue
        try:
            docs.extend(loader.load())
        except Exception as e:
            print(e, one_file)
    return docs

# 目标文件夹
tar_dir = [
    "./vg_data"
]


# 加载目标文件
docs = []
for dir_path in tar_dir:
    # copy_pdfs_to_root("/root/mydb/storage")
    docs.extend(get_text(dir_path))

# 对文本进行分块，每块为500个字符，重叠字符数为150
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=150)
split_docs = text_splitter.split_documents(docs)

# 加载开源词向量模型
embeddings = HuggingFaceEmbeddings(model_name="./sentence-transformer")

# 构建向量数据库
# 定义持久化路径
persist_directory = './data_base/vector_db/chroma'
# 加载数据库
vectordb = Chroma.from_documents(
    documents=split_docs,
    embedding=embeddings,
    persist_directory=persist_directory  # 允许我们将persist_directory目录保存到磁盘上
)
# 将加载的向量数据库持久化到磁盘上
vectordb.persist()