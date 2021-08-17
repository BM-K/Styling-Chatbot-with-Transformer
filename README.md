# Subject
Language Style과 감정에 따른 챗봇 답변 변화 모델 

# Environment
- Ubuntu 18.04.1 LTS
- Force RTX 2080 Ti
- Python 3.6.8
- Pytorch 1.2.0

# Model Structure
<img src = 'https://user-images.githubusercontent.com/55969260/85481753-65f30480-b5fd-11ea-8449-ba55a7ffd404.png'>

# Result
<img src = 'https://user-images.githubusercontent.com/55969260/85481920-c5e9ab00-b5fd-11ea-8010-27aa603fd78e.png'>

# Evaluation
Human Evaluation을 통해 성능 평가를 진행하였다. 질문 총 20개를 통해 얻어낸 답변으로 성능 옵션 5개인 매우 잘함(5점), 잘함(4점), 보통(3점), 못함(2점), 매우 못함(1점)에 대해 평가를 진행하였다. 총 27명의 평가자가 진행하였으며, 20개의 질문에 대한 평균 점수는 3.035점이 나왔다. 각각의 옵션에 대해 높은 평가 비율을 보인 질문과 답변 쌍은 [표 3]과 같으며 매우 잘함 55.6%라는 말은 27명 중 약 15명의 사람이 본 질문과 답변 쌍에 대해 매우 잘함이라 평가한 것이다. 
<br> <br>
<img src = 'https://user-images.githubusercontent.com/55969260/85482220-fcbfc100-b5fd-11ea-9601-4627175415b1.png'>

# Conclusion
본 논문에서는 사용자의 감정을 분석하고 출력의 종결 어미나 보조용언과 연결어미가 합쳐져 표현된 형태소를 찾아 Language Style을 변경시킴으로써 챗봇이 실제 사 람처럼 답변하게끔 하는 모델을 제시하였다. Human Evaluation을 통해 5개의 옵션을 두고 성능을 평가하였 다. 
<br> 본 연구를 통해 기존보다 사람에 가까운 답변을 하는 챗봇을 만들어 사용자 경험 측면에서의 만족도를 높이었다. Language Style을 구성하는 것은 말투만이 아니기에 향후 추가 학습을 통해 어휘, 묘사 등의 변화를 주어서 챗봇의 성능을 개선하고자 한다.

# References
[1] Ilya Sutskever, Oriol Vinyals and Quoc V.Le, “Sequence to Sequence Learning with Neural Networks”, In proceedings of Neural Information Processing System, pp. 3104-3112, 2014 
<br>[2] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N.Gomez, Lukasz Kaiser and Illia Polosukhin, “Attention Is All You Need”, In proceedings of Neural Information Processing System, pp. 5998-6008, 2017 
<br>[3] Junjie Tin*, Zixun Chen, kELAI Zhou and Chonhyuan Yu, “A Deep Learning Based Chatbot System for Campus Psychological Therapy”, arXiv preprint arXiv:1910.06707, pp. 0-31, 2019 
<br>[4] Jacob Devlin, Ming-Wei Chang, Kenton Lee and Kristina Toutanova, “BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding”, North American Chapter of the Association for Computational Linguistics, pp. 4171-4186, 2019 
