import evaluate

# 加载 F1 指标，F1 包含精确度和召回率
f1_metric = evaluate.load("f1")

# 定义预测和真实标签
predictions = [0, 1, 2, 1, 0, 2]
references = [0, 1, 2, 0, 0, 1]

# 计算每个标签的精确度、召回率和 F1 分数
result = f1_metric.compute(predictions=predictions, references=references, average=None)

# 显示每个标签的结果
labels = sorted(set(references))  # 提取所有标签
for i, label in enumerate(labels):
    # precision = result['precision'][i]
    # recall = result['recall'][i]
    f1 = result['f1'][i]
    print(f"Label {label}:")
    # print(f"  Precision: {precision}")
    # print(f"  Recall: {recall}")
    print(f"  F1 Score: {f1}")
