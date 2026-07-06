import time
import numpy as np
import uhd

CENTER_FREQ = 1e9
SAMPLE_RATE = 1e6
TONE_FREQ = 100e3

CAPTURE_DURATION = 0.08
TX_DURATION = 0.04
TX_DELAY = 0.02

TX_AMPLITUDE = 0.2
TX_GAIN_CH0 = 10
TX_GAIN_CH1 = 0
RX_GAIN = 40

TX_CHANS = [0, 1]
RX_CHANS = [0, 1]

usrp = uhd.usrp.MultiUSRP()

for ch in TX_CHANS:
    usrp.set_tx_rate(SAMPLE_RATE, ch)
    usrp.set_tx_freq(uhd.types.TuneRequest(CENTER_FREQ), ch)
    usrp.set_tx_antenna("TX/RX", ch)

usrp.set_tx_gain(TX_GAIN_CH0, 0)
usrp.set_tx_gain(TX_GAIN_CH1, 1)

for ch in RX_CHANS:
    usrp.set_rx_rate(SAMPLE_RATE, ch)
    usrp.set_rx_freq(uhd.types.TuneRequest(CENTER_FREQ), ch)
    usrp.set_rx_gain(RX_GAIN, ch)
    usrp.set_rx_antenna("RX2", ch)

time.sleep(1.0)

print("\nDual RX using 2TX/2RX mode")
print("TX0: tone on TX/RXA")
print("TX1: zeros/silent on TX/RXB")
print("RX0: RX2A")
print("RX1: RX2B")
print()

num_rx_samps = int(CAPTURE_DURATION * SAMPLE_RATE)
num_tx_samps = int(TX_DURATION * SAMPLE_RATE)

t = np.arange(num_tx_samps) / SAMPLE_RATE

tx0 = TX_AMPLITUDE * np.exp(1j * 2 * np.pi * TONE_FREQ * t)
tx1 = np.zeros(num_tx_samps, dtype=np.complex64)

tx_signal = np.vstack([
    tx0.astype(np.complex64),
    tx1.astype(np.complex64)
])

tx_args = uhd.usrp.StreamArgs("fc32", "sc16")
tx_args.channels = TX_CHANS
tx_streamer = usrp.get_tx_stream(tx_args)

rx_args = uhd.usrp.StreamArgs("fc32", "sc16")
rx_args.channels = RX_CHANS
rx_streamer = usrp.get_rx_stream(rx_args)

rx_buffer = np.zeros((2, num_rx_samps), dtype=np.complex64)
recv_buffer = np.zeros((2, 4096), dtype=np.complex64)

rx_md = uhd.types.RXMetadata()
tx_md = uhd.types.TXMetadata()

usrp.set_time_now(uhd.types.TimeSpec(0.0))
time.sleep(0.1)

rx_start_time = usrp.get_time_now().get_real_secs() + 0.2
tx_start_time = rx_start_time + TX_DELAY

stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.num_done)
stream_cmd.num_samps = num_rx_samps
stream_cmd.stream_now = False
stream_cmd.time_spec = uhd.types.TimeSpec(rx_start_time)
rx_streamer.issue_stream_cmd(stream_cmd)

tx_md.start_of_burst = True
tx_md.end_of_burst = False
tx_md.has_time_spec = True
tx_md.time_spec = uhd.types.TimeSpec(tx_start_time)

samples_sent = 0

while samples_sent < num_tx_samps:
    chunk = tx_signal[:, samples_sent:samples_sent + 4096]

    if samples_sent + chunk.shape[1] >= num_tx_samps:
        tx_md.end_of_burst = True

    sent = tx_streamer.send(chunk, tx_md, timeout=2.0)
    samples_sent += sent

    tx_md.start_of_burst = False
    tx_md.has_time_spec = False

samples_received = 0

while samples_received < num_rx_samps:
    samps = rx_streamer.recv(recv_buffer, rx_md, timeout=3.0)

    if rx_md.error_code != uhd.types.RXMetadataErrorCode.none:
        print("RX error:", rx_md.strerror())
        break

    end = min(samples_received + samps, num_rx_samps)
    rx_buffer[:, samples_received:end] = recv_buffer[:, :end - samples_received]
    samples_received = end

rx0 = rx_buffer[0, :samples_received]
rx1 = rx_buffer[1, :samples_received]

tx_start_sample = int(TX_DELAY * SAMPLE_RATE)
tx_end_sample = tx_start_sample + num_tx_samps

rx0_tone = rx0[tx_start_sample:tx_end_sample]
rx1_tone = rx1[tx_start_sample:tx_end_sample]

n = np.arange(len(rx0_tone))
ref = np.exp(-1j * 2 * np.pi * TONE_FREQ * n / SAMPLE_RATE)

tone0 = np.mean(rx0_tone * ref)
tone1 = np.mean(rx1_tone * ref)

amp0_dbfs = 20 * np.log10(np.abs(tone0) + 1e-12)
amp1_dbfs = 20 * np.log10(np.abs(tone1) + 1e-12)

phase0 = np.angle(tone0, deg=True)
phase1 = np.angle(tone1, deg=True)
phase_diff = np.angle(tone1 * np.conj(tone0), deg=True)

print(f"Samples sent      : {samples_sent}")
print(f"Samples received  : {samples_received}")
print(f"CH0 tone level    : {amp0_dbfs:.2f} dBFS")
print(f"CH1 tone level    : {amp1_dbfs:.2f} dBFS")
print(f"CH1 - CH0 level   : {amp1_dbfs - amp0_dbfs:.2f} dB")
print(f"CH0 phase         : {phase0:.2f} deg")
print(f"CH1 phase         : {phase1:.2f} deg")
print(f"CH1 - CH0 phase   : {phase_diff:.2f} deg")

np.save("dual_rx_ch0.npy", rx0)
np.save("dual_rx_ch1.npy", rx1)

print("Saved dual_rx_ch0.npy and dual_rx_ch1.npy")
