#这个库包的用法就是对中文文本进行预处理
import nltk
import os
from .unit import Unit
import mzcn as mz
import jieba
from opencc import OpenCC

"""
去除中文停用词
"""
#读取包内置的停用表
def getFile():
    #初始化一个空的停用词表
    stop=[]
    file_stop_url=mz.__file__.split('\\')
    cr=str('\\'.join(file_stop_url[0:-1]+['preprocessors','units','']))

    #获取停用词表
    tmp=cr+'stopwords.txt'
    with open(tmp,'r',encoding='utf-8') as f :
        lines = f.readlines()  # lines是list类型
        for line in lines:
            lline  = line.strip()     # line 是str类型,strip 去掉\n换行符
            if lline not in stop:
                stop.append(lline)        # 将stop 是列表形式

    #获取哈工大停用词表
    #tmp=cr+'哈工大停用词表.txt'
    #with open(tmp,'r',encoding='utf-8') as f :
    #    lines = f.readlines()  # lines是list类型
    #    for line in lines:
    #        lline  = line.strip()     # line 是str类型,strip 去掉\n换行符
    #        if lline not in stop:
    #            stop.append(lline)        # 将stop 是列表形式

    #获取四川大学停用词表
    #tmp=cr+'四川大学机器智能实验室停用词库.txt'
    #with open(tmp,'r',encoding='utf-8') as f :
    #    lines = f.readlines()  # lines是list类型
    #    for line in lines:
    #        lline  = line.strip()     # line 是str类型,strip 去掉\n换行符
    #        if lline not in stop:
    #            stop.append(lline)        # 将stop 是列表形式

    #获取中文停用词表
    #tmp='中文停用词库.txt'
    #with open(tmp,'r',encoding='utf-8') as f :
    #    lines = f.readlines()  # lines是list类型
    #    for line in lines:
    #        lline  = line.strip()     # line 是str类型,strip 去掉\n换行符
    #        if lline not in stop:
    #            stop.append(lline)        # 将stop 是列表形式
    return stop


class ChineseStopRemoval(Unit):
    """
    Process unit to remove stop words.
    去除中文文本的停用词

    Example:
        >>> unit = ChineseStopRemoval()
        >>> unit.transform(['我', '的', '心里','只有','搞钱'])
        '搞钱'
        >>> type(unit.stopwords)
        <class 'list'>
    """

    def __init__(self):
        """Initialization."""
        self._stop = getFile()

   

    def transform(self, input_: str) -> str:
        """
        移除分词列表里面的停用词

        :param input_: 待移除停用词的分词列表.
        
        :return cr: 去除停用词后的列表，并且将其以' '为间隔连接成字符串，准备接下来的预测.
        """
        pre_input_=jieba.lcut(input_)
        tmp=[token
                for token
                in pre_input_
                if token not in self._stop]
        cr=str(''.join(tmp))
        return cr 

    @property
    def stopwords(self) -> list:
        """
        Get stopwords based on language.

        :params lang: language code.
        :return: list of stop words.
        """
        return self._stop

"""
去除文本中的空格
"""
def process(pre_data):     #定义函数
    content = pre_data.replace(' ','')   # 去掉文本中的空格
    return content

class ChineseRemoveBlack(Unit):
    """
    去除文本中的空格.

    Example:
        >>>contents = '   大家好， 欢迎一起来学习文本的空格   去除   ！' 
        >>>unit = ChineseRemoveBlack()
        >>>unit.transform(contents)
        '大家好，欢迎一起来学习文本的空格去除！'
    """

    def transform(self, input_: str) -> str:
        """
        去除文本中的空格
        :param input_: list of tokenized tokens.
        :param lang: language code for stopwords.

        :return tokens: list of tokenized tokens without stopwords.
        """
        return process(input_)

import re

"""
去除掉文本中的表情符号
"""
def clear_character(sentence):    
    pattern = re.compile("[^\u4e00-\u9fa5^,^.^!^a-z^A-Z^0-9]")  #只保留中英文、数字和符号，去掉其他东西
    #若只保留中英文和数字，则替换为[^\u4e00-\u9fa5^a-z^A-Z^0-9]
    line=re.sub(pattern,'',sentence)  #把文本中匹配到的字符替换成空字符
    new_sentence=''.join(line.split())    #去除空白
    return new_sentence

class ChineseEmotion(Unit):
    """
    Process unit to remove stop words.

    Example:
        >>> unit = ChineseEmotion()
        >>> unit.transform('现在听着音乐,duo rui mi,很开心*_*')
        现在听着音乐,duoruimi,很开心
        
    """


    def transform(self, input_: str) -> str:
        """
        Remove stopwords from list of tokenized tokens.

        :param input_: list of tokenized tokens.
        :param lang: language code for stopwords.

        :return tokens: list of tokenized tokens without stopwords.
        """
        return clear_character(input_)


"""
中文简体繁体互相转化
"""
def Simplified(sentence):
    c = OpenCC('t2s')
    new_sentence = c.convert(sentence)   # 繁体转为简体
    return new_sentence


class ChineseSimplified(Unit):
    """
    Process unit to remove stop words.

    Example:
        >>> unit = ChineseSimplified()
        >>> unit.transform('你现在读的这里是简体，這裡是繁體，能看懂嗎？')
        你现在读的这里是简体，这里是繁体，能看懂吗？
    """

    def transform(self, input_: str) -> str:
        """
        Remove stopwords from list of tokenized tokens.

        :param input_: list of tokenized tokens.
        :param lang: language code for stopwords.

        :return tokens: list of tokenized tokens without stopwords.
        """
        return Simplified(input_)


#让文本只保留汉字
def is_chinese(uchar):
    if uchar >= u'\u4e00' and uchar <= u'\u9fa5':  # 判断一个uchar是否是汉字
        return True
    else:
        return False
 
def allcontents(contents):
    content = ''
    for i in contents:
        if is_chinese(i):
            content = content+i
    content1=str(' '.join(jieba.lcut(content)))
    return content1


class IsChinese(Unit):
    """
    Process unit to remove stop words.

    Example:
        >>> unit = IsChinese()
        >>> unit.transform('1,2,3...我们开始吧， 加油！')
        我们开始吧加油
    """

    def transform(self, input_: str) -> str:
        """
        只保留文本语料

        :param input_: 待处理的文本
        

        :return output: 得到只保留中文文字的文本
        """
        return allcontents(input_)


#这里就是mzcn的TF版本里面直接抄的，用于做中文文本的预处理
#这个是检查是否具有中文文本标记
def is_chinese_char(cp):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean
    # characters, despite its name. The modern Korean Hangul alphabet is a
    # different block, as is Japanese Hiragana and Katakana. Those alphabets
    # are used to write space-separated words, so they are not treated
    # specially and handled like the all of the other languages.
    return (0x4E00 <= cp <= 0x9FFF) or \
           (0x3400 <= cp <= 0x4DBF) or \
           (0x20000 <= cp <= 0x2A6DF) or \
           (0x2A700 <= cp <= 0x2B73F) or \
           (0x2B740 <= cp <= 0x2B81F) or \
           (0x2B820 <= cp <= 0x2CEAF) or \
           (0xF900 <= cp <= 0xFAFF) or \
           (0x2F800 <= cp <= 0x2FA1F)

#这里是为了Tokenize做处理，所以得出的结果
class ChineseTokenizeDemo(Unit):
    """预处理具有中文字符的语句"""

    def transform(self, input_: str) -> str:
        """
        Process input data from raw terms to processed text.

        :input_: raw textual input.

        :return output: text with at least one blank between adjacent
                        Chinese tokens.
        """
        output = []
        for char in input_:
            cp = ord(char)
            if is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
                #print('='*20)
            else:
                output.append(char)
        return "".join(output)
    

