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
RX_GAIN = 40
TX_GAINS = [0, 5, 10, 15, 20]

TX_CHAN = 1
RX_CHAN = 1
TX_ANT = "TX/RX"
RX_ANT = "RX2"

usrp = uhd.usrp.MultiUSRP()

usrp.set_tx_rate(SAMPLE_RATE, TX_CHAN)
usrp.set_rx_rate(SAMPLE_RATE, RX_CHAN)
usrp.set_tx_freq(uhd.types.TuneRequest(CENTER_FREQ), TX_CHAN)
usrp.set_rx_freq(uhd.types.TuneRequest(CENTER_FREQ), RX_CHAN)
usrp.set_rx_gain(RX_GAIN, RX_CHAN)
usrp.set_tx_antenna(TX_ANT, TX_CHAN)
usrp.set_rx_antenna(RX_ANT, RX_CHAN)

time.sleep(1.0)

tx_args = uhd.usrp.StreamArgs("fc32", "sc16")
tx_args.channels = [TX_CHAN]
tx_streamer = usrp.get_tx_stream(tx_args)

rx_args = uhd.usrp.StreamArgs("fc32", "sc16")
rx_args.channels = [RX_CHAN]
rx_streamer = usrp.get_rx_stream(rx_args)

num_rx_samps = int(CAPTURE_DURATION * SAMPLE_RATE)
num_tx_samps = int(TX_DURATION * SAMPLE_RATE)

t = np.arange(num_tx_samps) / SAMPLE_RATE
tx_signal = TX_AMPLITUDE * np.exp(1j * 2 * np.pi * TONE_FREQ * t)
tx_signal = tx_signal.astype(np.complex64)

tx_start_sample = int(TX_DELAY * SAMPLE_RATE)
tx_end_sample = tx_start_sample + num_tx_samps

print("\nTX gain sweep")
print("Use enough attenuation between TX and RX.")
print(f"RX gain fixed: {RX_GAIN} dB")
print("------------------------------------")
print("TX gain dB | Tone dBFS | Max dBFS")
print("------------------------------------")

results = []

for tx_gain in TX_GAINS:
    usrp.set_tx_gain(tx_gain, TX_CHAN)

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
    rx_tone_region = rx[tx_start_sample:tx_end_sample]

    n = np.arange(len(rx_tone_region))
    ref = np.exp(-1j * 2 * np.pi * TONE_FREQ * n / SAMPLE_RATE)

    tone_amp = np.abs(np.mean(rx_tone_region * ref))
    tone_dbfs = 20 * np.log10(tone_amp + 1e-12)

    max_amp = np.max(np.abs(rx_tone_region))
    max_dbfs = 20 * np.log10(max_amp + 1e-12)

    results.append((tx_gain, tone_dbfs, max_dbfs))

    print(f"{tx_gain:9.1f} | {tone_dbfs:9.1f} | {max_dbfs:8.1f}")

np.savetxt(
    "tx_gain_sweep.csv",
    np.array(results),
    delimiter=",",
    header="tx_gain_db,tone_dbfs,max_dbfs",
    comments=""
)

print("------------------------------------")
print("Saved tx_gain_sweep.csv")
