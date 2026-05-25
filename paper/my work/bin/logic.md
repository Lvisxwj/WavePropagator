1. 阅读C:\Users\xwj\Desktop\study\Machine Learning\cassi重构\src\paper\analysis的所有md，对照最新的代码C:\Users\xwj\Desktop\study\Machine Learning\cassi重构\src\version2，了解完整的pipeline。/init生成一个claude.md便于你快速查找

2. 所有的名字对照内容在C:\Users\xwj\Desktop\study\Machine Learning\cassi重构\src\paper\my work\name_mapping.md。需要以这个为基准

3. 根据了解到的pipeline和所有的md信息，完成逐个完成三个md的编写来助力论文写作

problem.md：需要涵盖（意思是这些要覆盖到，可以多出，不能少于，宁多误少，便于我后续删改）对问题的描述，Observation，gap analysis，key insight，assumptions等等。需要的部分可以借鉴参考学习顶会论文，在C:\Users\xwj\Desktop\study\Machine Learning\cassi重构\src\paper\reference_paper_pdf

algorithm.md：分两部分，第一部分需要详细，完整，严谨的数学推导，具体内容查看之前的md，不能遗漏或者偷懒。第二部分，工程上的对应，比如swap初始化了几个语意场，其对应的数学公式是那个，需要标注出来

architecture.md：三大part的所有主要组件（part1，2，3），以及隶属，或者交叉的（比如estimator的噪声对swap的影响）的详细pipeline，绘图颜色风格，比如part的背景，很多个组件可能有都layernorm，softplus，conv1x1这种，应该分别统一一个颜色。我挑选了一些：
#f3f2f7
#3a155c
反正就是想几个配色使之协调


根据代码，对每个part的每个组件，分别按照part name分别放到一个.py文件里，比如swap总的就是swap.py，隶属swap的AdaSpec，就叫swap-AdaSpec.py，然后根据这些py文件编写成ONNX格式的内容，文件名一样，一起放到一个文件夹里。完整的包括conponent graph with inputs, outputs, interactions。这部分的内容是方便我画图的。完成之后对照name_mapping.md检查是否遗漏

逐个完成