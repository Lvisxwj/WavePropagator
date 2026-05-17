dataset.py 增加新功能
1 记录数据集的band数，nc改成软编码，因为后续可能用到新数据集
2 记录每张图片的波长范围到 [data_name]_info.json 里
3 记录每个band的均值到 [data_name]_bands.json 里
比如目前我用的是CAVE_1024,就应该是CAVE_1024_bands.json
这两个jason都放在dataset/data_name/下面

physics.py 修改WAVELENGTHS_CAVE_28 成 WAVELENGTHS
里面的内容应该用CAVE_1024_bands.json的内容

train.py 应该保留新模型的mask选项，因为后续可能会加入新的mask，而不是在模型里面硬编码成maskA
增加打印功能，开始训练的第一轮前会打印开始时间，24小时制，如5.4.2:19; 每50轮的时候同样打印当时的时间，估计结束的时间，和前面的格式对齐

同时对于dataset的新功能，应先检查 [data_name]的两个json存不存在，不做重复的工作


loss.py 增加频域幅度差异 和 在展现哪个频率段差异最大的分层分析的指标（好像叫径向频率分布对比）可以展示低频段差异和高频的差异，并且在training的时候连同psrn ssim一起打印出来

viz.py 增加原图的频域图和重构后的频域图 展现频域差异的图，都合并到show all的那个函数里面，并保留我自己写的main

总之loss和viz我是想体现我模型毕竟是频域的，一些体现指标和有助于后续完善模型的指标和思路，你帮我想一些，记住上述说的修改可能是局限的，可能其他.py的一些函数也需要一定的修改！！！最后生成一个5.4_solution.md，先不做代码修改
















