import random
import math
# RANSAC implementation for text rectification
class TextRectifier:
    def __init__(self, confidance, sigma, max_iter, lower_case = False, ignore_inliers = True) -> None:
        self.prediction = None
        self.inliers = 0
        self.confidence = confidance
        self.lower_case = lower_case
        self.max_iter = max_iter
        self.sigma = sigma
        self.sample_size = 0
        self.do_ransac = False
        self.ignore_inliers = ignore_inliers
        pass

    def feed_data(self, text_list:list, score_list:list) -> None:
        self.raw_data = {'text': text_list, 'score': score_list}
        self.samples = []

        for s, p in zip(text_list, score_list):
            if self.lower_case:
                s = s.lower()

            # Replace character other than english letters to space
            s = ''.join([i if ord(i) < 128 else ' ' for i in s])

            self.samples.append({'T': s, 'P': p})

        # Detect duplicate samples
        for i in range(len(self.samples)):
            for j in range(i + 1, len(self.samples)):
                if self.samples[i]['T'] == self.samples[j]['T']:
                    self.do_ransac = True

        self.sample_size = len(self.samples)

    def calc_fitness(self, sample, prediction):
        return 1-sample['P'] if sample['T'] == prediction else sample['P']
    
    def fit(self) -> None:
        if self.sample_size == 0:
            return

        if self.do_ransac == False: # Maximum likelyhood
            self.prediction = max(self.samples, key=lambda x:x['P'])['T']
            self.inliers = [i for i in range(self.sample_size)]
            return

        else:   
            i = 0
            self.inliers = []
            while i < self.max_iter:
                sample_index = random.sample(range(self.sample_size), 1)
                sample = self.samples[sample_index[0]]

                prediction = sample['T']

                total_inlier = []
                for index in range(self.sample_size):
                    t = self.samples[index]
                    if self.calc_fitness(t, prediction) < self.sigma:
                        total_inlier.append(index)

                if len(total_inlier) > len(self.inliers):
                    if len(total_inlier) == self.sample_size:
                        self.inliers = total_inlier
                        self.prediction = prediction
                        break
                    self.max_iter = math.log(1 - self.confidence) / math.log(1 - pow(len(total_inlier) / (self.sample_size), 2))
                    self.inliers = total_inlier
                    self.prediction = prediction

                i += 1

        return
    
    def get_rectified_text(self):
        ret = []
        if self.ignore_inliers:
            return [self.prediction for i in range(self.sample_size)]

        for i in range(self.sample_size):
            if i in self.inliers:
                ret.append(self.prediction)
            else:
                ret.append(self.raw_data['text'][i])

        return ret