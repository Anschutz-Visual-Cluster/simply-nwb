import numpy as np
from matplotlib import pylab as plt
from matplotlib import widgets as wid
from matplotlib import lines
from simply_nwb.pipeline.util.saccade_gui import removeArrowKeyBindings
# Code sourced from: https://github.com/jbhunt/myphdlib/blob/7a6dd65fa410e985853027767d95010872aff505/myphdlib/extensions/matplotlib.py


class SaccadeDirectionLabelingGUI(object):
    """
    """

    def __init__(
        self,
        figsize=(5, 5)
        ):
        """
        """

        #
        self.xlim = (-1, 1)
        self.ylim = (-1, 1)
        self.labels = (
            'Left',
            'Right',
            'Noise',
            'Unscored'
        )

        #
        self.fig, self.ax = plt.subplots()
        self.fig.subplots_adjust(left=0.35)
        self.fig.set_figwidth(figsize[0])
        self.fig.set_figheight(figsize[1])
        self.wave = lines.Line2D([0], [0], color='k')
        self.cross = {
            'vertical': lines.Line2D([0, 0], self.ax.get_ylim(), color='k', alpha=0.3),
            'horizontal': lines.Line2D(self.ax.get_xlim(), [0, 0], color='k', alpha=0.3)
        }
        self.ax.add_line(self.wave)
        for line in self.cross.values():
            self.ax.add_line(line)

        #
        self.sampleIndex = 0

        # Checkbox panel
        self.checkboxPanel = wid.RadioButtons(
            plt.axes([0.02, 0.5, 0.2, 0.2]),
            labels=self.labels,
            active=3
        )
        self.checkboxPanel.on_clicked(self.onCheckboxClicked)

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
            plt.axes([0.02, 0.2, 0.15, 0.05]),
            label='Exit',
            color='white',
            hovercolor='grey'
        )
        self.exitButton.on_clicked(self.onExitButtonClicked)
        removeArrowKeyBindings()
        self.fig.canvas.callbacks.connect('key_press_event', self.onKeyPress)

        return

    def inputSamples(self, samples, gain=1.3, randomizeSamples=False):
        """
        """

        #
        self.xTrain = samples
        if randomizeSamples:
            np.random.shuffle(self.xTrain)
        nSamples, nFeatures = self.xTrain.shape
        self.y = np.full([self.xTrain.shape[0], 1], np.nan)

        #
        self.ylim = np.array([0, nFeatures - 1])
        self.xlim = np.array([
            np.nanmin(self.xTrain) * gain,
            np.nanmax(self.xTrain) * gain
        ])
        if np.any(np.isnan(self.xlim)):
            raise Exception('Fuck')

        #
        if nFeatures % 2 == 0:
            center = nFeatures / 2
        else:
            center = (nFeatures - 1) / 2
        self.cross['vertical'].set_data([0, 0], self.ylim)
        self.cross['horizontal'].set_data(self.xlim, [center, center])

        #
        self.updatePlot()
        plt.show()

        return

    def updatePlot(self):
        """
        """

        wave = np.take(self.xTrain, self.sampleIndex, mode='wrap', axis=0)
        self.wave.set_data(wave, np.arange(wave.size))
        self.ax.set_ylim(self.ylim)
        self.ax.set_xlim(self.xlim)
        self.fig.canvas.draw()

        return

    def updateCheckboxPanel(self):
        """
        """

        currentLabel = np.take(self.y, self.sampleIndex, axis=0, mode='wrap')
        if currentLabel == -1:
            self.checkboxPanel.set_active(0)
        elif currentLabel == +1:
            self.checkboxPanel.set_active(1)
        elif currentLabel == 0:
            self.checkboxPanel.set_active(2)
        elif np.isnan(currentLabel):
            self.checkboxPanel.set_active(3)

        return

    def onCheckboxClicked(self, buttonLabel):
        """
        """

        checkboxIndex = np.where(np.array(self.labels) == buttonLabel)[0].item()
        newLabel = np.array([-1, 1, 0, np.nan])[checkboxIndex]
        sampleIndex = np.take(np.arange(self.y.size), self.sampleIndex, mode='wrap')
        self.y[sampleIndex] = newLabel

        return

    def cycleRadioButtons(self, direction=-1):
        """
        """

        buttonLabel = self.checkboxPanel.value_selected
        currentButtonIndex = np.where(np.array(self.labels) == buttonLabel)[0].item()
        nextButtonIndex = np.take(np.arange(4), currentButtonIndex + direction, mode='wrap')
        self.checkboxPanel.set_active(nextButtonIndex)

        return

    def onNextButtonClicked(self, event):
        """
        """

        self.sampleIndex += 1
        self.updatePlot()
        self.updateCheckboxPanel()

        return

    def onPreviousButtonClicked(self, event):
        """
        """

        self.sampleIndex -= 1
        self.updatePlot()
        self.updateCheckboxPanel()

        return

    def onExitButtonClicked(self, event):
        """
        """

        plt.close(self.fig)

        return

    def isRunning(self):
        """
        """

        return plt.fignum_exists(self.fig.number)

    def updatePlot(self):
        """
        """

        wave = np.take(self.xTrain, self.sampleIndex, mode='wrap', axis=0)
        self.wave.set_data(wave, np.arange(wave.size))
        self.ax.set_ylim(self.ylim)
        self.ax.set_xlim(self.xlim)
        self.fig.canvas.draw()

        return

    def onKeyPress(self, event):
        """
        """

        if event.key in ('up', 'down', 'left', 'right'):
            if event.key == 'up':
                self.cycleRadioButtons(-1)
            if event.key == 'down':
                self.cycleRadioButtons(+1)
            self.updateCheckboxPanel()

        if event.key == 'enter':
            self.sampleIndex += 1
            self.updatePlot()
            self.updateCheckboxPanel()

        return

    @property
    def trainingData(self):
        """
        """

        mask = np.invert(np.isnan(self.y.flatten()))
        X = self.xTrain[mask, :]
        y = self.y[mask, :]

        return X, y
