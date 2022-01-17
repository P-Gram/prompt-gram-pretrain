eval "$(pyenv init -)"
#eval "$(pyenv virtualenv-init -)"
pyenv activate macbert-env
#--use_jieba 使用jieba分词  不使用 删除此参数
CUDA_VISIBLE_DEVICES="2" python run_chinese_ref.py \
    --file_name=./data/corpus.txt \
    --ltp=/Users/qiwang/company/Document/pretrain-models/ltp \
    --bert=/Users/qiwang/company/Document/pretrain-models/bert-base-chinese\
    --save_path=./data/ref.txt \
    --ltp_batch_size=1000 \
    --use_jieba 1 \
#    --user_dict=user_dict.txt
