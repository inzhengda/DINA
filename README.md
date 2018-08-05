实验数据来源：
http://staff.ustc.edu.cn/%7Eqiliuql/data/math2015.rar.  
数据集已经添加在math2015这个文件夹下面，里面包含3个数据集，FrcSub是经典数据集，分数减法的计算，Math1、Math2是某市部分高中生的数学考试记录。数据集具体详情见里面的说明文档。  

针对该数据集的代码文件，为DINA_math.py。数据集里面的文件格式转换成了csv方便处理，文件代码的注释和功能还在完善添加。  

DINA.py是针对JAVA大项目的代码  


在认知诊断评估中，Deterministic Inputs，Noisy“And”gate model( DINA 模型)旨在对学生多维知识点掌握程度进行建模分析，它能够在精准建模学生学习状态的同时保证了较好的可解释性，近年来广受学者关注和研究。 具体地，DINA 结合了Q 矩阵作为试题的先验知识，将学生的潜在学习状态描述成一个多维知识点掌握向量，同时引入题目的猜测和失误参数，以准确地在多维知识层面诊断学生的认知学习状态。   
知识点掌握程度：DINA模型是一种离散型的认知诊断模型，它将学生的能力描述为一个多维知识点技能向量αｊ ＝ （αｊ１，αｊ２，…，αｊｋ)，且定义试题ｉ关联一组知识点技能ｑｉ ＝（ｑｉ１，ｑｉ２，…ｑ ）ｉｋ。则学生ｊ对于试题ｉ的掌握情况定义δｊｉ为：

当有I个学生，J道题目和K个知识点时，DINA模型中的潜在作答变量ηij可以表示为:  
![avatar](https://raw.githubusercontent.com/inzhengda/ImageRepo/master/p1.png)

其中αik = 1 或0 表示学生i 掌握或没有掌握知识点k，学生的潜在能力矩阵α = { α1，α2
，…，αI} '，Q 矩阵Q ={ qjk}J * K。故ηij反映了学生能力是否足够答对该题。在引入失误率s 和猜测率g 两种题目参数后，实际响应矩阵X{Xij}I*J的概率模型为:
![avatar](https://raw.githubusercontent.com/inzhengda/ImageRepo/master/p2.png)

![avatar](https://raw.githubusercontent.com/inzhengda/ImageRepo/master/p3.png)

![avatar](https://raw.githubusercontent.com/inzhengda/ImageRepo/master/p4.png)

![avatar](https://raw.githubusercontent.com/inzhengda/ImageRepo/master/p5.png)

