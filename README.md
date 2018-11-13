# 瑞金医院MMC人工智能辅助构建知识图谱大赛
- 仅供同道中人学习使用

- [比赛说明](https://tianchi.aliyun.com/competition/information.htm?spm=5176.100067.5678.2.3cbb24c0PwM849&raceId=231687)
- [原始数据_测试数据a_测试数据b](https://pan.baidu.com/s/1X_5iRsoGxVbUPFW3zKTz-w)  提取码：rf0g

## 第一赛季 
  - 成绩 0.7282

## 预处理
- 清洗文本 .txt `clean.py.cli_clean_txt()`

- 分析标注文件 `clean.py.cli_parse_anns()`

- 添加标注 .txt.tag `./clean.py.cli_tag_lines()`

## 使用说明
- 修改对应代码

```
train.py 

...

# available models
import bilstm_crf
import att_bilstm
import att_cnn
import att_bilstm_crf

    ...

        # TODO: set the model to use
	self.model = att_cnn
        # TODO: save model fp
        self.model_fp = './model/att_bilstm.h5'

    ...

```

### 训练词向量
- `./c2v.py ./train`

### 训练
- `./train.py train ./train`

### 校验
- `./train.py validate ./model/bilstm_crf_model.h5 ./train`

### 测试
- `./train.py test ./model/bilstm_crf_model.h5 ./test`

## 说明
- 可用模型: `bilstm_crf_model.h5, att_cnn.h5`
- BiLSTM+CRF 的评测结果优于 ATT+CNN , 但训练速度ATT+CNN更快(10x)
- 可能在更大的训练集上，ATT+CNN会更好

## 感谢
- [attention\_keras.py](https://github.com/bojone/attention)
- [自然语言处理中的自注意力机制](https://www.cnblogs.com/robert-dlut/p/8638283.html)
