from modelscope import snapshot_download

# local_dir 指定你要存放模型的文件夹
model_path = snapshot_download(
    model_id="BAAI/bge-reranker-v2-m3",
    local_dir="./models/bge-reranker-v2-m3",
)
print("模型下载完成，路径：", model_path)