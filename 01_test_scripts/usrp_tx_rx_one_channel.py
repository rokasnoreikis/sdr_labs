import time
import numpy as np
import uhd

CENTER_FREQ = 1e9
SAMPLE_RATE = 1e6
TONE_FREQ = 100e3

CAPTURE_DURATION = 0.08
TX_DURATION = 0.04
TX_DELAY = 0.02

TX_AMPLITUDE = 0.5

TX_GAIN = 10
RX_GAIN = 60

TX_CHAN = 0
RX_CHAN = 0

TX_ANT = "TX/RX"
RX_ANT = "RX2"

usrp = uhd.usrp.MultiUSRP()

usrp.set_tx_rate(SAMPLE_RATE, TX_CHAN)
usrp.set_rx_rate(SAMPLE_RATE, RX_CHAN)

usrp.set_tx_freq(uhd.types.TuneRequest(CENTER_FREQ), TX_CHAN)
usrp.set_rx_freq(uhd.types.TuneRequest(CENTER_FREQ), RX_CHAN)

usrp.set_tx_gain(TX_GAIN, TX_CHAN)
usrp.set_rx_gain(RX_GAIN, RX_CHAN)

usrp.set_tx_antenna(TX_ANT, TX_CHAN)
usrp.set_rx_antenna(RX_ANT, RX_CHAN)

time.sleep(1.0)

print("\n===== USRP Configuration =====")
print(f"TX channel        : {TX_CHAN}")
print(f"RX channel        : {RX_CHAN}")
print(f"TX antenna        : {usrp.get_tx_antenna(TX_CHAN)}")
print(f"RX antenna        : {usrp.get_rx_antenna(RX_CHAN)}")
print(f"TX gain           : {usrp.get_tx_gain(TX_CHAN):.1f} dB")
print(f"RX gain           : {usrp.get_rx_gain(RX_CHAN):.1f} dB")
print(f"TX rate           : {usrp.get_tx_rate(TX_CHAN):.0f} S/s")
print(f"RX rate           : {usrp.get_rx_rate(RX_CHAN):.0f} S/s")
print(f"TX freq           : {usrp.get_tx_freq(TX_CHAN)/1e6:.6f} MHz")
print(f"RX freq           : {usrp.get_rx_freq(RX_CHAN)/1e6:.6f} MHz")
print(f"Tone freq         : {TONE_FREQ/1e3:.3f} kHz")
print("==============================\n")

num_rx_samps = int(CAPTURE_DURATION * SAMPLE_RATE)
num_tx_samps = int(TX_DURATION * SAMPLE_RATE)

t = np.arange(num_tx_samps) / SAMPLE_RATE
tx_signal = TX_AMPLITUDE * np.exp(1j * 2 * np.pi * TONE_FREQ * t)
tx_signal = tx_signal.astype(np.complex64)

tx_args = uhd.usrp.StreamArgs("fc32", "sc16")
tx_args.channels = [TX_CHAN]
tx_streamer = usrp.get_tx_stream(tx_args)

rx_args = uhd.usrp.StreamArgs("fc32", "sc16")
rx_args.channels = [RX_CHAN]
rx_streamer = usrp.get_rx_stream(rx_args)

rx_buffer = np.zeros(num_rx_samps, dtype=np.complex64)
recv_buffer = np.zeros(4096, dtype=np.complex64)

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
    chunk = tx_signal[samples_sent:samples_sent + 4096]

    if samples_sent + len(chunk) >= num_tx_samps:
        tx_md.end_of_burst = True

    sent = tx_streamer.send(chunk, tx_md, timeout=2.0)
    samples_sent += sent

    tx_md.start_of_burst = False
    tx_md.has_time_spec = False

print(f"Sent {samples_sent} samples")

samples_received = 0

while samples_received < num_rx_samps:
    samps = rx_streamer.recv(recv_buffer, rx_md, timeout=3.0)

    if rx_md.error_code != uhd.types.RXMetadataErrorCode.none:
        print("RX error:", rx_md.strerror())
        break

    end = min(samples_received + samps, num_rx_samps)
    rx_buffer[samples_received:end] = recv_buffer[:end - samples_received]
    samples_received = end

rx = rx_buffer[:samples_received]

print(f"Received {samples_received} samples")

mean_amp = np.mean(np.abs(rx))
max_amp = np.max(np.abs(rx))
peak_dbfs = 20 * np.log10(max_amp + 1e-12)

n = np.arange(len(rx))
ref = np.exp(-1j * 2 * np.pi * TONE_FREQ * n / SAMPLE_RATE)
tone_amp = np.abs(np.mean(rx * ref))
tone_dbfs = 20 * np.log10(tone_amp + 1e-12)

print(f"RX mean amplitude          : {mean_amp:.6f}")
print(f"RX max amplitude           : {max_amp:.6f}")
print(f"RX peak                    : {peak_dbfs:.1f} dBFS")
print(f"Detected 100 kHz amplitude : {tone_amp:.6f}")
print(f"Detected 100 kHz level     : {tone_dbfs:.1f} dBFS")

np.save("rx_iq.npy", rx)
print("Saved rx_iq.npy")
