import os

import numpy as np

STATES = {'B', 'I', 'E', 'S'}


class HMM(object):
    def __init__(self, train_data, test_data, test_label):
        self.trd = train_data
        self.ted = test_data
        self.tel = test_label
        self.STATES = {'B', 'I', 'E', 'S'}
        self.states = []  # 实际状态序列
        self.observes = []  # 观测序列
        self.trans_mat = {}  # trans_mat[status][status] = int
        self.emit_mat = {}  # emit_mat[status][observe] = int
        self.init_vec = {}  # init_vec[status] = int
        self.state_count = {}  # state_count[status] = int
        # 初始化矩阵
        for state in STATES:
            self.trans_mat[state] = {}
            for target in STATES:
                self.trans_mat[state][target] = 0.0  # 状态转移矩阵， trans_mat[state1][target]表示训练集中由state转移到target的次数。
            self.emit_mat[state] = {}  # 观测矩阵， emit_mat[state][char]表示训练集中单字char被标注为state的次数
            self.init_vec[state] = 0  # 初始状态分布向量， init_vec[state]表示状态state在训练集中出现的次数
            self.state_count[state] = 0  # 状态统计向量
        self.seg_stop_words = {" ", "，", "。", "“", "”", '“', "？", "！", "：", "《", "》", "、", "；", "·", "‘ ", "’",
                               "──", ",", ".", "?", "!", "`", "~", "@", "#", "$", "%", "^", "&", "*", "(", ")", "-",
                               "_", "+", "=", "[", "]", "{", "}", '"', "'", "<", ">", "\\", "|" "\r", "\n", "\t", "「",
                               "」"}
        pra = self.trd
        for line in pra:
            line = line.strip()
            if not line:
                continue
            # 获取观测序列
            for i in range(len(line)):
                if line[i] not in self.seg_stop_words:
                    self.observes.append(line[i])
            words = line.split(" ")
            # 获取实际状态序列
            for word in words:
                if word not in self.seg_stop_words:
                    tag = []
                    if len(word) == 1:
                        tag = ['S']
                    elif len(word) == 2:
                        tag = ['B', 'E']
                    else:
                        num = len(word) - 2
                        tag.append('B')
                        tag.extend(['I'] * num)
                        tag.append('E')
                    self.states.extend(tag)
            # 计数，记频率
            if len(self.observes) >= len(self.states):
                for i in range(len(self.states)):
                    if i == 0:
                        self.init_vec[self.states[0]] += 1
                        self.state_count[self.states[0]] += 1
                    else:
                        self.trans_mat[self.states[i - 1]][self.states[i]] += 1
                        self.state_count[self.states[i]] += 1
                    if self.observes[i] not in self.emit_mat[self.states[i]]:
                        self.emit_mat[self.states[i]][self.observes[i]] = 1
                    else:
                        self.emit_mat[self.states[i]][self.observes[i]] += 1
            else:
                pass

    def compute_start_prob(self):  # 初始状态矩阵
        init_vec1 = {}
        asum = sum(self.init_vec.values())
        for key in self.init_vec:
            init_vec1[key] = float(self.init_vec[key]) / asum
        return init_vec1

    def compute_transition(self):  # 转移矩阵
        trans_mat1 = {}
        default = max(self.state_count.values())
        for key1 in self.trans_mat:
            trans_mat1[key1] = {}
            for key2 in self.trans_mat[key1]:
                if self.state_count[key1] == 0:
                    trans_mat1[key1][key2] = float(self.trans_mat[key1][key2]) / default
                else:
                    trans_mat1[key1][key2] = float(self.trans_mat[key1][key2]) / self.state_count[key1]
        return trans_mat1

    def compute_emission(self):  # 发射矩阵
        emit_mat1 = {}
        default = max(self.state_count.values())
        for key1 in self.emit_mat:
            emit_mat1[key1] = {}
            for key2 in self.emit_mat[key1]:
                if self.state_count[key1] == 0:
                    emit_mat1[key1][key2] = float(self.emit_mat[key1][key2]) / default
                else:
                    emit_mat1[key1][key2] = float(self.emit_mat[key1][key2]) / self.state_count[key1]
        return emit_mat1

    def viterbi_decoding(self):
        EPS: float = 0.00001
        o = 0
        init_vec1 = self.compute_start_prob()
        trans_mat1 = self.compute_transition()
        emit_mat1 = self.compute_emission()
        sentence = self.ted  # 1493
        sequence = sentence[o]
        tab = [{}]  # 动态规划表
        path = {}

        for state in STATES:
            tab[0][state] = init_vec1[state] * emit_mat1[state].get(sequence[0], EPS)
            path[state] = [state]
        # 创建动态搜索表
        for t in range(1, len(sequence)):
            tab.append({})
            new_path = {}
            for state1 in STATES:
                items = []
                for state2 in STATES:
                    if tab[t - 1][state2] == 0:
                        if tab[t - 1][state2] == 0:
                            prob = -3.14e+100#tab[t - 1][state2] + trans_mat1[state2].get(state1, EPS) + emit_mat1[state1]['end']
                        #else:
                            #prob = tab[t - 1][state2] + trans_mat1[state2].get(state1, EPS) + emit_mat1[state1]['begin']
                    else:
                        prob = tab[t - 1][state2] * trans_mat1[state2].get(state1, EPS) * emit_mat1[state1].get(
                            sequence[t - 1], EPS)
                    items.append((prob, state2))
                best = max(items)
                tab[t][state1] = best[0]
                new_path[state1] = path[best[1]] + [state1]
            path = new_path
            # 搜索最优路径
        prob, state = max([(tab[len(sequence) - 1][state], state) for state in STATES])
        print(len(path[state]))
        return prob, state, path

    def word_segmentation(self):
        prob, state, path = self.viterbi_decoding()
        sentence = self.ted  # 1493
        for o in range(0, len(sentence)-1):
            sequence = sentence[o]
            segment = ''
            for i in range(len(path[state])):
                j = path[state][i]
                if j == 'B':
                    segment = segment + sequence[i]
                else:
                    if j == 'I':
                        segment = segment + sequence[i]
                    else:
                        if j == 'E':
                            segment = segment + sequence[i] + ' '
                        else:
                            segment = segment + sequence[i] + ' '
        return segment


def main():
    train_data = open(os.path.join('D:\pythonProject', 'WordSegmentation', 'train.txt'),
                      encoding='utf-8').read().splitlines()
    test_data = open(os.path.join('D:\pythonProject', 'WordSegmentation', 'test.txt'),
                     encoding='utf-8').read().splitlines()
    test_label = open(os.path.join('D:\pythonProject', 'WordSegmentation', 'test_gold.txt'),
                      encoding='utf-8').read().splitlines()
    hmm = HMM(train_data, test_data, test_label)
    segment = hmm.word_segmentation()
    with open("my_prediction.txt", "w", encoding='utf-8') as f:
        f.write(segment)
    # Step 1: Learning

    # Step 2: Decoding

    # Step 3: Calculate F1 score

    # Step 4: Write your prediction to a file (the format should be the same as test_gold.txt)
    # You need to convert BIES to the format like test_gold.txt (call hmm.word_segmentation() here)
    # fout = open('my_prediction.txt', 'w', encoding='utf-8')


main()
