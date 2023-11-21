import jieba
from tqdm import tqdm
import logging
from jieba import analyse
jieba.setLogLevel(log_level=logging.INFO)

words=''
with open("lawcontent.txt", 'r', encoding='utf-8') as f:
        datas=f.readlines()
        for i,data in enumerate(tqdm(datas)):   
            c=data.strip().split('\t')
            query_content=c[0]
            text = query_content
            keywords = analyse.extract_tags(text, topK=10, withWeight=True, allowPOS=('n', 'v'))
            

            for i in range(len(keywords)):
              word=keywords[i][0]
              l=word+'\n'
              words+=l

with open("lawpinyin.txt", 'w', encoding='utf-8') as f:
  for i in words:
    f.write(i)

