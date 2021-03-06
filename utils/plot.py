# Taken from https://github.com/noagarcia/visdom-tutorial

from visdom import Visdom
import numpy as np



class VisdomLinePlotter:
    """Plots to Visdom"""
    def __init__(self, env_name='main'):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}
    def plot(self, var_name, split_name, title_name, x_label, x, y):
        """
        var_name: variable name (e.g. loss, acc)
        split_name: split name (e.g. train, val)
        title_name: titles of the graph (e.g. Classification Accuracy)
        x: x axis value (e.g. epoch number)
        y: y axis value (e.g. epoch loss)
        """
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line([y,y], [x,x], env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                xlabel=x_label,
                ylabel=var_name
            ))
        else:
            self.viz.line([y], [x], env=self.env, win=self.plots[var_name], name=split_name, update = 'append')

if __name__ == '__main__':
    test = VisdomLinePlotter()
    for i in range(10):
        test.plot('value', 'function', 'testing', 'loop', i, i*2)