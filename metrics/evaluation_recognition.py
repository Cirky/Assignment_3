import math
import numpy as np
import matplotlib.pyplot as plt
import json
import cv2

class Evaluation:

    def compute_rank1(self, Y, y):
        classes = np.unique(sorted(y))
        count_all = 0
        count_correct = 0
        # print(classes)
        # print(Y)
        for cla1 in classes:
            idx1 = y == cla1
            if (list(idx1).count(True)) <= 1:
                continue
            # Compute only for cases where there is more than one sample:
            Y1 = Y[idx1 == True, :]
            Y1[Y1 == 0] = math.inf

            for y1 in Y1:  # basically vprasa a je tisti z najmanjso razdaljo tudi pravi class
                s = np.argsort(y1)  # s je tabela indexov po vrsti od najmanjse vrednosti naprej
                min_index = s[0]
                min_value_class = idx1[min_index]
                count_all += 1
                # print(min_value)
                if min_value_class:
                    count_correct += 1
        return count_correct / count_all * 100

    # Add your own metrics here, such as rank5, (all ranks), CMC plot, ROC, ...

    def compute_rank_n(self, Y, y, n):
        # First loop over classes in order to select the closest for each class.
        classes = np.unique(sorted(y))
        count_all = 0
        count_correct = 0

        for cla1 in classes:
            idx1 = y == cla1
            if (list(idx1).count(True)) <= 1:
                continue
            Y1 = Y[idx1 == True, :]
            Y1[Y1 == 0] = math.inf


            for y1 in Y1:  # basically vprasa a je tisti z najmanjso razdaljo tudi pravi class
                s = np.argsort(y1)  # s je tabela indexov po vrsti od najmanjse vrednosti naprej
                min_index = s[0]
                min_value_class = idx1[min_index]
                count_all += 1
                # print(min_value)
                neki = []
                for cla2 in classes:
                    # Select the closest that is higher than zero:
                    idx2 = y == cla2
                    if (list(idx2).count(True)) <= 1:
                        continue
                    Y2 = y1[idx2 == True]
                    Y2[Y2 == 0] = math.inf
                    min_val = np.min(np.array(Y2))
                    neki.append((min_val, cla2))
                neki = sorted(neki, key=lambda x: x[0])
                for i in range(n):
                    if neki[i][1] == cla1:
                        count_correct += 1
                        break

        return count_correct / count_all * 100

    def plot_CMC(self, Y, y, name):
        ranks = []
        rank_values = []
        value = 0
        for rank in range(1, 101):
            ranks.append(rank)
            if value >= 100:
                rank_values.append(100)
            else:
                value = self.compute_rank_n(Y, y, rank)
                rank_values.append(value)

        plt.plot(ranks, rank_values)
        plt.ylabel('Accuracy [%]')
        plt.xlabel('Rank')
        plt.title(name)
        plt.show()
        plt.savefig('results_data/cmc_p2p_yolo.png')

