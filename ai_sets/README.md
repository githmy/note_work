# 1. 运行环境python3.6
# 2. 项目依赖包
pip install -r requirements.txt

# 3. 训练
# 自动加载 ../nocode/input/testproject/conf/json_file.csv 的参数
# 自动加载 ../nocode/conf/testproject/json_file.json 的参数
python train.py --project testproject --jsonfile 10000num --model model_tmp

# 4. 训练输出,预测
# 模型位置 ../nocode/models/testproject/model_name/xx.pkl
# 否则模型位置 ../nocode/models/testproject/model_datetime/xx.pkl
python predict.py --project testproject --jsonfile 10000num --model model_tmp

# 5. 服务记录
# log文件 ../nocode/logs/testproject/json_file.log
# 输出文件 ../nocode/output/testproject/json_file.csv
python server.py -P 4445


# 6. Open a new terminal and now you can curl results from the server, for example:
curl -XGET localhost:4445/version
## 输入 vector vector unitid modelid和 网络类型，直接从train下找unitid文件，列名为setid，模型为unitid
## classtype, max_features, maxlen, memn, dropn
curl -XPOST 10.4.252.92:4445/process -d '{"project":4,"branch":5,"usertype":"general","userid":3}' | jq '.'
curl -XPOST 10.4.252.92:4445/process -d '{"project":4,"branch":5,"usertype":"admin","userid":0}' | jq '.'

curl -XPOST localhost:4445/predict -d '{"q":{"word_txt1":["我该吃什么药"],"word_txt2":["你该吃什么药"]},"project":"testproject","model":"model_20170921-170911","file":"10000num"}' | jq '.'
curl -XPOST localhost:4445/hard -d '{"q":{"word_txt1":["我该吃什么药"],"word_txt2":["你该吃什么药"]},"project":"testproject","model":"model_20170921-170911","file":"10000num"}' | jq '.'
curl -XPOST localhost:4445/parse -d '{"q":{"word_txt1":["我该吃什么药"],"word_txt2":["你该吃什么药"]},"project":"testproject","model":"model_20170921-170911","file":"10000num"}' | jq '.'


# 7. 访问形式：训练（立刻断开，周期性的请求结果），预测（立刻返回），检索（立刻返回），
# 8. 维护模式：系统文件存储，客户文件
# 9. log rsyslog kaflka

git
entityid,type,词
unitid,type,rpsuid,type,relation,type,rpsuid,type
unitid:{modeldefaultmodel:modelid,defaultnomodel:nomodelid,modelid: parameters,nomodelid: parameters}

# 10. 客户目录
配置文件的路径 + 项目编号 + 实例编号 + 用户编号