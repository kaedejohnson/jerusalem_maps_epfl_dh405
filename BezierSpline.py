import numpy as np
import matplotlib.pyplot as plt

class BezierSpline:
    def __init__(self, draw_samples = 8, curvature_samples = 20):
        self.draw_samples = draw_samples
        self.curvature_samples = curvature_samples

    def from_pca(self, pca_anchor_1, pca_anchor_2, std_var_factor = 1.0, using_component1 = 0, using_component2 = 0):
        self.points = [pca_anchor_1['Centroid'], pca_anchor_2['Centroid']]
        self.directions = [pca_anchor_1['PCA_Basis'][using_component1], pca_anchor_2['PCA_Basis'][using_component2]]
        self.weights = std_var_factor * np.array([pca_anchor_1['PCA_Expands'][using_component1], pca_anchor_2['PCA_Expands'][using_component2]])

        self.calc_control_points()

    def from_pdw(self, points: list, directions: list, weights: list):
        self.points = points
        self.directions = directions
        self.weights = weights

        self.calc_control_points()

    def calc_control_points(self):
        prev_d = np.array(self.directions[0])
        for i in range(1, len(self.directions)):
            _d = np.array(self.directions[i])
            dot = np.dot(prev_d, _d)
            if dot < 0:
                _d = -_d
            prev_d = _d
            self.directions[i] = _d

        self.control_points = []
        for seg_id in range(0, len(self.points)-1):
            _p = []
            _p.append(self.points[seg_id])
            _p.append([self.points[seg_id][0] + self.weights[seg_id] * self.directions[seg_id][0], self.points[seg_id][1] + self.weights[seg_id] * self.directions[seg_id][1]])
            _p.append([self.points[seg_id + 1][0] - self.weights[seg_id] * self.directions[seg_id + 1][0], self.points[seg_id + 1][1] - self.weights[seg_id] * self.directions[seg_id + 1][1]])
            _p.append(self.points[seg_id + 1])
            self.control_points.append(np.array(_p))

    def B_nx(self, n, i, x):
        if i > n:
            return 0
        elif i == 0:
            return (1-x)**n
        elif i == 1:
            return n*x*( (1-x)**(n-1))
        return self.B_nx(n-1, i, x)*(1-x)+self.B_nx(n-1, i-1, x)*x

    def get_value(self, p, canshu):
        sumx = 0.
        sumy = 0.
        length = len(p)-1
        for i in range(0, len(p)):
            sumx += (self.B_nx(length, i, canshu) * p[i][0])
            sumy += (self.B_nx(length, i, canshu) * p[i][1])
        return sumx, sumy

    def get_newxy(self, p,x):
        xx = [0] * len(x)
        yy = [0] * len(x)
        for i in range(0, len(x)):
            a, b = self.get_value(p, x[i])
            xx[i] = a
            yy[i] = b
        return xx, yy

    def draw(self):
        x = np.linspace(0, 1, self.draw_samples)
        xx = []
        yy = []

        for i in range(0, len(self.control_points)):
            p = self.control_points[i]
            xx2, yy2 = self.get_newxy(p, x)
            xx += xx2
            yy += yy2

        plt.plot(xx, yy, 'r', linewidth=1)
        plt.scatter((xx)[:], (yy)[:], 1, "blue")
        plt.show()

    def get_as_polygon(self, num_pts = 8):
        x = np.linspace(0, 1, num_pts)
        xx = []
        yy = []

        for i in range(0, len(self.control_points)):
            p = self.control_points[i]
            xx2, yy2 = self.get_newxy(p, x)
            xx += xx2
            yy += yy2

        pts1 = [(xx[i], yy[i]) for i in range(len(xx))]
        pts2 = [(xx[i], yy[i]) for i in range(len(xx)-2, 0, -1)]
        return pts1 + pts2


    def _bezier_curve(self, t, control_points):
        n = len(control_points) - 1
        b = [(1 - t)**(n - i) * t**i for i in range(n + 1)]
        return np.dot(b, control_points)

    def _bezier_derivative(self, t, control_points, degree):
        n = len(control_points)
        b = [n * (control_points[i + 1] - control_points[i]) for i in range(n - 1)]
        if degree == 1:
            return self._bezier_curve(t, b)
        else:
            return self._bezier_derivative(t, b, degree - 1)

    def curvature(self, t, seg_id):
        first_derivative = self._bezier_derivative(t, self.control_points[seg_id], 1)
        second_derivative = self._bezier_derivative(t, self.control_points[seg_id], 2)
        
        numerator = np.linalg.norm(np.cross(first_derivative, second_derivative))
        denominator = np.linalg.norm(first_derivative)**3
        
        return numerator / denominator
    
    def get_max_curvature_for_seg(self, seg_id, num_samples = 20):
        self.curvature_samples = num_samples
        max_curvature = 0
        for t in np.linspace(0, 1, self.curvature_samples):
            curvature_value = self.curvature(t, seg_id)
            if curvature_value > max_curvature:
                max_curvature = curvature_value
        return max_curvature

    def get_max_curvature(self, num_samples = 20):
        self.curvature_samples = num_samples
        max_curvature = 0
        for seg_id in range(0, len(self.control_points)):
            for t in np.linspace(0, 1, self.curvature_samples):
                curvature_value = self.curvature(t, seg_id)
                if curvature_value > max_curvature:
                    max_curvature = curvature_value
        return max_curvature
    
    def get_min_curvature_for_seg(self, seg_id, num_samples = 20):
        self.curvature_samples = num_samples
        min_curvature = 100000000
        for t in np.linspace(0, 1, self.curvature_samples):
            curvature_value = self.curvature(t, seg_id)
            if curvature_value < min_curvature:
                min_curvature = curvature_value
        return min_curvature
    
    def get_min_curvature(self, num_samples = 20):
        self.curvature_samples = num_samples
        min_curvature = 100000000
        for seg_id in range(0, len(self.control_points)):
            for t in np.linspace(0, 1, self.curvature_samples):
                curvature_value = self.curvature(t, seg_id)
                if curvature_value < min_curvature:
                    min_curvature = curvature_value
        return min_curvature
    
    def get_control_points(self, seg_id):
        return self.control_points[seg_id]
    
    def get_control_seg_length(self, seg_id, control_seg_id):
        return np.linalg.norm(self.control_points[seg_id][control_seg_id] - self.control_points[seg_id][control_seg_id + 1])