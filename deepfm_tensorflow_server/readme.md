主要代码功能：
DeepFM: 基于TensorFlow平台实现deepfm模型算法，并且将训练好的模型数据保存在本地
estimator： deepfm模型预测器，可以从本地加载deepfm模型数据，重建graph计算依赖图，在内存中还原deepfm模型
1. run_web_server： 1. 接受web_test客户端发来的post请求，解析后写入redis队列中，通知run_model_server执行模型预测任务
                    2. 一旦统计mode_server 去执行某个用户的ctr预估任务后，异步等待，查询redis，获取该用户的计算结果
2. run_model_server：初始化时先实现estimator模型从本地加载到内存中，然后不断等待web_server发过来的用户预测任务，解析队列中的用户
                    特征数据，交给estimator完成ctr预估，将结果重新写入redis中，交由web_server 获取
3. web_test： 客户端，发起post请求，将某个用户的特征数据发送给web_server，并等待post的返回结果，获取用户最终的ctr预测结果


主要流程：
1. web_test 发起post api请求，请求参数中包含用户的相关特征
2. run_web_server 服务程序接收到post请求，获取用户的特征数据，将用户特征数据包装成message，作为一个模型预测任务，存入redis队列中，
    并不断查询redis 队列，等待该用户的预测结果
3. run_model_server: 加载了deepfm 的ctr预估模型，不断查询redis队列，一旦获取到web_server写入到队列中 包含用户特征数据的模型预测
    任务请求，立即从队列中取出json 字符串格式的用户特征数据，交给实际的deepfm模型分类器 deepfmestimator 进行预测
4. 模型预测的结果重新写入redis中，由web_server 程序获取，一旦web_serve 获取到后，通过post返回给web_test 客户端
5. web_test拿到post请求返回的结果，里面就是该用户id对应的ctr预估结果

启动流程：
1. 先启动run_model_server: 完成基于TensorFlow平台的deepfm模型加载到内存中
2. 接着启动flask web服务程序： run_web_server
3. 调用web_test里面的api 请求接口