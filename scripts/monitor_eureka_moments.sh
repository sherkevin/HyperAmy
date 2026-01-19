#!/bin/bash

# Eureka Moments 监控脚本
# 用于实时捕获Dynamic Weighting的"觉醒时刻"日志

LOG_FILE="test_vibe_search_experiment_run.log"
OUTPUT_FILE="eureka_moments.log"

echo "🚀 启动 Eureka Moments 监控..."
echo "------------------------------------------------------------"
echo "监控日志文件: ${LOG_FILE}"
echo "输出文件: ${OUTPUT_FILE}"
echo "筛选条件: Dynamic Weighting 且 W_emo >= 0.3"
echo "------------------------------------------------------------"
echo ""

# 创建输出文件（如果不存在）
touch "${OUTPUT_FILE}"

# 添加时间戳到输出文件
echo "============================================================" >> "${OUTPUT_FILE}"
echo "监控开始时间: $(date '+%Y-%m-%d %H:%M:%S')" >> "${OUTPUT_FILE}"
echo "============================================================" >> "${OUTPUT_FILE}"
echo "" >> "${OUTPUT_FILE}"

# 使用tail -f监控日志，过滤Dynamic Weighting相关的行
# 特别关注W_emo >= 0.3的情况（觉醒时刻）
tail -f "${LOG_FILE}" 2>/dev/null | while IFS= read -r line; do
    # 检查是否包含"Dynamic Weighting"
    if echo "$line" | grep -q "Dynamic Weighting"; then
        # 提取W_emo值（格式可能是W_emo=0.45或Final W_emo=0.45等）
        w_emo=$(echo "$line" | grep -oE '(W_emo|Final W_emo)=([0-9]+\.[0-9]+)' | grep -oE '[0-9]+\.[0-9]+' | head -1)
        
        if [ -n "$w_emo" ]; then
            # 检查W_emo是否 >= 0.3（使用bc进行浮点数比较）
            if (( $(echo "$w_emo >= 0.3" | bc -l) )); then
                # 同时输出到终端和文件
                echo "[$(date '+%H:%M:%S')] 🎉 觉醒时刻捕获: $line"
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] $line" >> "${OUTPUT_FILE}"
                
                # 如果W_emo >= 0.7，这是一个极端觉醒时刻
                if (( $(echo "$w_emo >= 0.7" | bc -l) )); then
                    echo "[$(date '+%H:%M:%S')] ⚡⚡⚡ 极端觉醒！W_emo=$w_emo >= 0.7！情绪权重完全接管！" | tee -a "${OUTPUT_FILE}"
                    echo "" >> "${OUTPUT_FILE}"
                fi
            fi
        else
            # 如果没有成功提取W_emo，但仍然包含Dynamic Weighting，也记录下来
            echo "[$(date '+%H:%M:%S')] 📊 Dynamic Weighting: $line"
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] $line" >> "${OUTPUT_FILE}"
        fi
    fi
    
    # 同时捕获其他关键指标：Iq, S_sem
    # 这些可以帮助我们理解"跷跷板效应"
    if echo "$line" | grep -qE "(Iq=|S_sem=)"; then
        # 只记录包含关键指标的行
        if echo "$line" | grep -q "Dynamic Weighting"; then
            # 已在上面处理，跳过
            continue
        fi
    fi
done
