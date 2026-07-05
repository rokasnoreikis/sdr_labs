import sys
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets

FS = 1e6
SKIP = 20000
MAX_SAMPLES = 100

iq = np.load("rx_iq.npy")
iq = iq[SKIP:SKIP + MAX_SAMPLES]

t_ms = np.arange(len(iq)) / FS * 1000

app = QtWidgets.QApplication(sys.argv)

win = pg.GraphicsLayoutWidget(title="Clean RX IQ Signal")
win.resize(1200, 500)

plot = win.addPlot(title="Received IQ Signal")
plot.plot(t_ms, iq.real, pen=pg.mkPen(width=2), name="I")
plot.plot(t_ms, iq.imag, pen=pg.mkPen(width=2), name="Q")

plot.setLabel("bottom", "Time", units="ms")
plot.setLabel("left", "Amplitude")
plot.showGrid(x=True, y=True)
plot.addLegend()

win.show()
sys.exit(app.exec())
