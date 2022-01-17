
# --gradient_accumulation_steps 8 

CUDA_VISIBLE_DEVICES="2,3" python -m torch.distributed.launch --nproc_per_node 2 --master_port 4377 run_mlm_wwm.py \
    --model_name_or_path bert-base-chinese \
    --train_file ./data/train.txt \
    --train_ref_file ./data/train.ref \
    --validation_file ./data/test.txt \
    --validation_ref_file ./data/test.ref \
    --do_train \
    --do_eval \
    --use_ngram \
    --output_dir ./result \
    --evaluation_strategy steps \
    --eval_step 10000 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --num_train_epochs 30 \
    --save_strategy steps \
    --save_step 10000 \
    --save_total_limit 10 \
    --mlm_probability 0.30 \
    --learning_rate 1e-4 \
    # --fp16
