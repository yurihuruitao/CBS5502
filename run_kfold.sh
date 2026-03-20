#!/bin/bash
# 5 折交叉验证：所有模型 × 5 折
# 用法: bash run_kfold.sh [模型名]
#   无参数: 运行全部模型
#   指定模型: bash run_kfold.sh bert

set -e
cd "$(dirname "$0")/src"

LOGDIR="../logs/kfold"
mkdir -p "$LOGDIR"

MODELS=("bilstm" "bert_frozen" "bert" "roberta" "deberta" "sbert")

# 如果指定了模型名，只运行该模型
if [ -n "$1" ]; then
    MODELS=("$1")
fi

for model in "${MODELS[@]}"; do
    for fold in 0 1 2 3 4; do
        echo "══════════════════════════════════════════"
        echo "  模型: $model  折: $fold"
        echo "══════════════════════════════════════════"
        logfile="$LOGDIR/${model}_fold${fold}.log"
        python3 -u "model_${model}.py" --fold "$fold" > >(tee "$logfile")
        echo ""
    done
done

echo ""
echo "═══════════════════════════════════"
echo "  全部训练完成，运行统计检验..."
echo "═══════════════════════════════════"
python3 statistical_tests.py
