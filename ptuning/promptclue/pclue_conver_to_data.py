import pandas as  pd
import json
def convert_json_to_csv(source_file, target_file):
    """将json文件转化为csv形式的文件。
       source_file:输入文件；
       target_file：转化后的文件
    """
    lines=open(source_file,'r').readlines()
    print("length of lines:",len(lines))
    input_list=[]
    output_list=[]
    answer_choices_list=[]
    type_list=[]
    for i, line in enumerate(lines):
        # {"input": "以下内容为真：“滁县地区专员张友道说:大都架到高处了”那么下面的陈述：“张友道对身边的官员说了话。”是真的,假的,或未知？\n答案：", "target": "未知", "answer_choices": ["真的", "假的", "未知"], "type": "nli"}
        # 1)获得字段值
        json_string=json.loads(line.strip())
        input_=json_string["input"].replace("\n", "_")
        output_=json_string["target"]
        answer_choices_=json_string.get("answer_choices",[])
        type_=json_string["type"]
        if i<10:print(i,"input:",input_,";output:",output_)
        # 2)添加到列表中
        input_list.append(input_)
        output_list.append(output_)
        answer_choices_list.append(answer_choices_)
        type_list.append(type_)

    # 3)写成pandas的dataframe，以csv进行保存
    df = pd.DataFrame({'input': input_list,
                       'target':output_list,
                       'answer_choices': answer_choices_list,
                       'type': type_list,
                       })
    df.to_csv(target_file,index=False)

# 请运行以下三行代码进行格式换行，如果你需要全量数据训练。
# 默认将只使用部分在线的示例数据进行训练。
source_file='pCLUE_train_1.json'
target_file='./pCLUE_train_1.csv'
convert_json_to_csv(source_file, target_file)
