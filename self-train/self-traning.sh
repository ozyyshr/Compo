python ./data/sample_subset.py 1

python run_summarization.py \
    --model_name_or_path ./bart-large-147 \
    --do_train \
    --do_eval \
    --do_predict \
    --train_file ./data/limited-147/train_1.json \
    --validation_file ./data/val.json \
    --test_file ./data/test.json \
    --text_column dialogue \
    --summary_column summary \
    --output_dir ./bart-large-147/sf-train-1 \
    --overwrite_output_dir \
    --per_device_train_batch_size=1 \
    --per_device_eval_batch_size=1 \
    --predict_with_generate


for i in 2 3 4 5
do

python ./data/sample_subset.py $i

python run_summarization.py \
    --model_name_or_path ./bart-large-147/sf-train-$(expr $i - 1) \
    --do_train \
    --do_eval \
    --do_predict \
    --train_file ./data/limited-147/train_$i.json \
    --validation_file ./data/val.json \
    --test_file ./data/test.json \
    --text_column dialogue \
    --summary_column summary \
    --output_dir ./bart-large-147/sf-train-$i \
    --overwrite_output_dir \
    --per_device_train_batch_size=1 \
    --per_device_eval_batch_size=1 \
    --predict_with_generate

done