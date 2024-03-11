accelerate launch --multi_gpu --num_processes=8 run_evals_accelerate.py \
    --model_args "pretrained=yentinglin/Taiwan-LLM-13B-v2.0-chat" \
    --override_batch_size 4 \
    --output_dir="./evals/" \
    --tasks tasks_examples/twllm_eval.txt \
    --custom_tasks "community_tasks/twllm_eval.py"
