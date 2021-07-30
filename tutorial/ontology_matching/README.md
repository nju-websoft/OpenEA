# 本体匹配

## 任务引入
由于本体采用分散式的开发方法，每个机构和个人都可以独立发布各自的本体，所以对于相交领域（甚至是相同领域），通常会存在多个本体。不同本体之间存在差异，即为异构性。本体匹配是消除本体间异构性的一种有效途径，可以为应用程序之间的交互建立互操作性，是知识融合的重要任务。



## 代码结构

### src

- `data_input.py`：从数据集本体文件onto.rdf中读取了所有本体，及每个本体的label。可以使用[rdflib](https://rdflib.readthedocs.io/en/stable/)从本体文件中读取并使用更多知识。
- `main.py`：程序入口。
- **`match.py`**：需要实现其中的`ontology_matching`方法，以完成本体匹配。
- `demo.py`：我们给出的一个基线方法以供参考。

### test

- `test.py`：根据ground truth对本体匹配的结果进行评估。

### datasets

其中101是完整的参考本体，而301、302、303和304都将与参考本体101进行对齐。

每个数据集中的`onto.rdf`即为本体文件，`refalign.rdf`是与参考本体101对齐的ground truth。



## 实验要求

修改`src/match.py`中的`ontology_matching`方法，实现一种优于基线方法的本体匹配方法。

### 基线介绍

基于直观的假设：label相似的两个本体更有可能对齐。

设置一个阈值，认为label文本相似度（编辑距离）超过阈值的两个本体是对齐的。

### 基线性能

| Dataset | Precision | Recall | F1-score |
| ------- | :-------- | ------ | -------- |
| 301     | 1.00      | 0.217  | 0.356    |
| 302     | 0.900     | 0.191  | 0.316    |
| 303     | 0.875     | 0.429  | 0.575    |
| 304     | 0.918     | 0.592  | 0.720    |



## 环境配置

安装rdflib：`pip install rdflib`

使用`src/demo.py`需要安装：

```
pip install numpy
pip install python-Levenshtein
```

