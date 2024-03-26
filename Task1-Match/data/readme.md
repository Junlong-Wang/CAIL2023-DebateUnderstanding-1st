# 论辩理解入门赛 输入输出格式说明

本赛道入门赛阶段所下发的数据集包括训练集 `train_*.jsonl` 和测试集 `test_*.jsonl`，其中训练集和测试集分别选自案由分类不重叠的案件，且涵盖刑事与民事案件。所有数据均以 JSON Line 格式存储，每个样本一个 JSON 字符串；文件中出现的 Unicode 字符（中文等）以转义方式存储，可用 `pandas` 或 `datasets` 包直接读取。

数据集中包含的文件有：

- `train_text.jsonl`：包含了裁判文书所有对于辩诉双方辩护全文的数据，共 $32125$ 条。每条数据包含的字段内容如下：
  - `sentence_id`：句子 ID
  - `text_id`：裁判文书 ID
  - `category`：刑事、民事案件分类
  - `chapter`：刑事罪名或民事案由所在章节
  - `crime`：具体的刑事罪名或民事案由
  - `position`：诉方（sc）与辩方（bc）标志
  - `sentence`：句子文本

- `train_entry.jsonl`：包含了 $9310$ 对裁判文书中的互动论点对，每条数据包含的字段内容如下：
  - `id`：论点对 ID
  - `text_id`：裁判文书 ID
  - `category`：刑事、民事案件分类
  - `chapter`：刑事罪名或民事案由所在章节
  - `crime`：具体的刑事罪名或民事案由
  - `sc`：诉方论点
  - `bc_x`（$x=1,2,3,4,5$）：候选辩方论点，共五句
  - `answer`：正确辩方论点编号

- `test_text.jsonl`：同下发数据中的 `train_text.jsonl` 格式完全一致，共 $9381$ 条数据；

- `test_entry.jsonl`：同下发数据中的 `train_entry.jsonl` 格式基本一致，包含了 $2490$ 对裁判文书中的互动论点对，但缺少相应的 `answer` 标签。



注：train0.7.json是伪标注数据
train2_7.json是添加伪标注数据后的数据