
使用wwm的方式进行mask，同时对被mask的单词随机显示其中的某个字符，比例为1/3, 整个单词进行损失,
ref分词时将单词长度大于7的单词按照长度4进行切分, 去除掉bert 80%mask 10%随机的机制，只采用mask和prompts


 --prompts_length 8 \
 --prompts_lr_length 12 \
 以上两个参数暂时没有作用 忽略