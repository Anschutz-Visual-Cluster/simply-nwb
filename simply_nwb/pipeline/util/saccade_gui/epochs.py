import numpy as np
from matplotlib import pylab as plt
from matplotlib import widgets as wid
from matplotlib import lines
from matplotlib.backend_bases import MouseButton
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl

from simply_nwb.pipeline.util.saccade_gui import removeArrowKeyBindings
# Code sourced from: https://github.com/jbhunt/myphdlib/blob/7a6dd65fa410e985853027767d95010872aff505/myphdlib/extensions/matplotlib.py


class SaccadeEpochLabelingGUI(object):
    """
    """

    def __init__(
            self,
            figsize=(6, 6)
    ):
        """
        """

        #
        self.xlim = (-1, 1)
        self.ylim = (-1, 1)
        self.t = np.array([])
        self.tc = 0

        #
        self.labels = (
            'Start',
            'Stop',
        )
        self.label = 'Start'

        #
        self.fig, self.ax = plt.subplots()
        self.fig.subplots_adjust(left=0.35)
        self.fig.set_figwidth(figsize[0])
        self.fig.set_figheight(figsize[1])
        self.lines = {
            'wave': lines.Line2D([0], [0], color='k', alpha=0.5),
            'start': lines.Line2D([0, 0], self.ax.get_ylim(), color='green', alpha=0.5),
            'stop': lines.Line2D([0, 0], self.ax.get_ylim(), color='red', alpha=0.5)
        }
        for line in self.lines.values():
            self.ax.add_line(line)
        self.line = self.lines['start']

        #
        self.sampleIndex = 0

        # Previous button
        self.previousButton = wid.Button(
            plt.axes([0.02, 0.4, 0.15, 0.05]),
            label='Previous',
            color='white',
            hovercolor='grey'
        )
        self.previousButton.on_clicked(self.onPreviousButtonClicked)

        # Next button
        self.nextButton = wid.Button(
            plt.axes([0.02, 0.3, 0.15, 0.05]),
            label='Next',
            color='white',
            hovercolor='grey'
        )
        self.nextButton.on_clicked(self.onNextButtonClicked)

        # Exit button
        self.exitButton = wid.Button(
            plt.axes([0.02, 0.2, 0.1, 0.1]),
            label='Exit',
            color='white',
            hovercolor='grey'
        )
        self.exitButton.on_clicked(self.onExitButtonClicked)

        # Checkbox panel
        self.checkboxPanel = wid.RadioButtons(
            plt.axes([0.02, 0.5, 0.2, 0.2]),
            labels=self.labels,
            active=0
        )
        self.checkboxPanel.on_clicked(self.onCheckboxClicked)

        #
        removeArrowKeyBindings()
        self.fig.canvas.callbacks.connect('key_press_event', self.onKeyPress)
        self.fig.canvas.callbacks.connect('button_press_event', self.onButtonPress)

    def updatePlot(self):
        """
        """

        wave = np.take(self.xTrain, self.sampleIndex, mode='wrap', axis=0)
        self.lines['wave'].set_data(self.t, wave)
        x1, x2 = np.take(self.y, self.sampleIndex, mode='wrap', axis=0)
        if np.isnan([x1, x2]).all():
            x1, x2 = self.tc, self.tc
        self.lines['start'].set_data([x1, x1], self.ylim)
        self.lines['stop'].set_data([x2, x2], self.ylim)
        self.fig.canvas.draw()

        return

    def resetAxesLimits(self):
        """
        """

        self.ax.set_ylim(self.ylim)
        self.ax.set_xlim(self.xlim)

        return

    def inputSamples(self, samples, labels, gain=1.5, randomizeSamples=False):
        """
        """

        #
        self.xTrain = samples
        if randomizeSamples:
            np.random.shuffle(self.xTrain)
        nSamples, nFeatures = self.xTrain.shape
        self.y = np.full([self.xTrain.shape[0], 2], np.nan)
        self.z = labels

        #
        self.xlim = np.array([0, nFeatures - 1])
        self.ylim = np.array([
            np.nanmin(self.xTrain) * gain,
            np.nanmax(self.xTrain) * gain
        ])

        #
        self.t = np.arange(nFeatures)
        self.tc = nFeatures / 2
        for key, line in self.lines.items():
            if key == 'wave':
                continue
            line.set_data([self.tc, self.tc], self.ylim)
        self.xlim = np.array([self.t.min(), self.t.max()])

        #
        self.resetAxesLimits()
        self.updatePlot()
        plt.show()

        return

    def onNextButtonClicked(self, event):
        """
        """

        self.sampleIndex += 1
        self.resetAxesLimits()
        self.updatePlot()
        self.label = 'Start'
        self.updateCheckboxPanel()

        return

    def onPreviousButtonClicked(self, event):
        """
        """

        self.sampleIndex -= 1
        self.resetAxesLimits()
        self.updatePlot()
        self.label = 'Start'
        self.updateCheckboxPanel()

        return

    def onExitButtonClicked(self, event):
        """
        """

        plt.close(self.fig)

        return

    def updateCheckboxPanel(self):
        """
        """

        if self.label == 'Start':
            self.checkboxPanel.set_active(0)
        else:
            self.checkboxPanel.set_active(1)

        return

    def onCheckboxClicked(self, buttonLabel):
        """
        """

        if buttonLabel == 'Start':
            self.line = self.lines['start']
            self.label = 'Start'
        else:
            self.line = self.lines['stop']
            self.label = 'Stop'

        return

    def onKeyPress(self, event):
        """
        """

        # Move the epoch boundary left or right
        clicked = False
        if event.key in ('left', 'shift+left', 'right', 'shift+right'):
            clicked = True

        if clicked:
            if event.key in ('left', 'right'):
                offset = 0.05
            elif event.key in ('shift+left', 'shift+right'):
                offset = 1
            x, y = self.line.get_data()
            if 'left' in event.key:
                xp = [
                    x[0] - offset,
                    x[1] - offset
                ]
                self.line.set_data(xp, y)
            elif 'right' in event.key:
                xp = [
                    x[0] + offset,
                    x[1] + offset
                ]
                self.line.set_data(xp, y)
            self.y[self.sampleIndex, 0] = self.lines['start'].get_data()[0][0]
            self.y[self.sampleIndex, 1] = self.lines['stop'].get_data()[0][0]
            self.updatePlot()

        # Toggle epoch boundaries
        clicked = False
        if event.key in ('up', 'down'):
            clicked = True

        if clicked:
            if self.label == 'Start':
                self.label = 'Stop'
                self.updateCheckboxPanel()
            elif self.label == 'Stop':
                self.label = 'Start'
                self.updateCheckboxPanel()

    def onButtonPress(self, event):
        """
        """

        # Move the epoch boundary where a click was made
        clicked = False
        if hasattr(event, 'button'):
            clicked = True

        if clicked:
            if event.button == 1 and event.key == 'shift':
                xf = event.xdata
                self.line.set_data([xf, xf], self.ylim)
                self.y[self.sampleIndex, 0] = self.lines['start'].get_data()[0][0]
                self.y[self.sampleIndex, 1] = self.lines['stop'].get_data()[0][0]
                self.updatePlot()

        return

    def isRunning(self):
        """
        """

        return plt.fignum_exists(self.fig.number)

    @property
    def labeledSamplesMask(self):
        return np.invert(np.isnan(self.y).all(axis=1))

    @property
    def trainingData(self):
        """
        """

        X = self.xTrain[self.labeledSamplesMask, :]
        y = np.around(self.y[self.labeledSamplesMask, :] - self.tc, 3)
        z = self.z[self.labeledSamplesMask]

        return X, y, z


class MarkerPlacer():
    """
    """

    def __init__(self, image=None, ax=None, fig=None, color='r'):
        """
        """

        #
        self.markers = None
        self.points = list()
        if ax is None and fig is None:
            self.fig, self.ax = plt.subplots()
        else:
            self.fig, self.ax = fig, ax
        if image is not None:
            self.image = self.ax.imshow(image, cmap='binary_r')
        else:
            self.image = None
        self.color = color
        self.fig.show()
        self.fig.canvas.callbacks.connect('button_press_event', self.onClick)
        self.fig.canvas.callbacks.connect('key_press_event', self.onPress)

        return

    def drawPoints(self):
        """
        """

        #
        if self.markers is not None:
            self.markers.remove()

        #
        x = [point[0] for point in self.points]
        y = [point[1] for point in self.points]
        self.markers = self.ax.scatter(x, y, color=self.color)

        #
        self.fig.canvas.draw()

        return

    def onClick(self, event):
        """
        """

        if event.button == MouseButton.LEFT and event.key == 'shift':
            point = (event.xdata, event.ydata)
            self.points.append(point)
        self.drawPoints()

        return

    def onPress(self, event):
        """
        """

        if event.key == 'ctrl+z':
            if self.points:
                point = self.points.pop()
        self.drawPoints()

        return


def placeMarkers(image=None):
    """
    """

    return
