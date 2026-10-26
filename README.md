# microbit_learns

A BBC micro:bit v2 that learns your hand gestures, trains on itself (no PC involved), and then flips your slides. You tilt the board a dozen times per gesture, it trains a small neural network on that, and from then on a tilt right sends Right arrow and a tilt left sends Left arrow to the laptop with the slides. The weights are born on the nRF52833, from your gestures.

This is the demo for a 60 minute talk at a Rust meetup. It is written for Rust programmers, not ML people, so it is six short files in no_std Rust that you can read top to bottom. No framework, no magic.

## Hardware

A micro:bit v2 and a USB cable. That's it. The accelerometer (an LSM303AGR) is already on the board.

## Build and flash

The toolchain comes from `rust-toolchain.toml` (stable, plus the `thumbv7em-none-eabihf` target). The flasher is probe-rs:

```bash
cargo install probe-rs-tools
cargo run --release
```

`cargo run` builds, flashes over USB and stays attached to show the debug output. That output is plain text over RTT (through the probe, not the serial port). A panic shows up there too, with file and line.

WSL note: the board belongs to Windows, so hand it to WSL first with usbipd-win. It shows up as VID:PID 0d28:0204:

```bash
usbipd list                          # find the busid of 0d28:0204
usbipd attach --wsl --busid <busid>
```

The busid changes with the USB port, so look it up each time. After that WSL sees a CMSIS-DAP probe and a serial port (`/dev/ttyACM0`). For driving slides the serial port has to be on a real Linux machine though, see below.

## The demo script

Reset the board (or flash it) and it starts recording. The four gestures go in a fixed order, and the display shows the icon of the one it wants:

| Order | Gesture | What you do                   | Icon         |
|-------|---------|-------------------------------|--------------|
| 1     | rest    | hold still, at any angle      | one dot      |
| 2     | next    | tilt right                    | right arrow  |
| 3     | prev    | tilt left                     | left arrow   |
| 4     | shake   | quick tilts side to side      | checkerboard |

The board reads the accelerometer every 20 ms and always keeps the last 640 ms (32 samples). Button A saves that window as one example of the current gesture. So: do the tilt, press A as it ends. Every press adds a dot to the display so you can count. Twelve per gesture, after that A does nothing. Button B moves on to the next gesture (it is ignored while the current one has zero examples). B on the fourth gesture starts training.

Training is 200 passes over the 48 windows. The display draws a spiral from the corner to the center, one pixel per 8 passes. When the center lights up, the thinking is over and the board is in work mode, until it is reset.

It predicts every 200 ms and, when at least 75 percent sure, shows the icon of what it thinks you are doing. The buttons do nothing now (press them anyway, I won't judge you). Reset starts over with fresh weights.

## Driving the slides

The board pretends to be a keyboard. Same trick as the 2024 meetup, where example 5 of <https://github.com/walruscraft/rust-examples-microbit-2024> sent PS/2 mouse codes over the UART. I must say I still like that micro:bit as a mouse demo better, check it out, it is fun. But this time it is just two keys instead of a mouse. On the Linux machine with the slides:

```bash
sudo inputattach --ps2serkbd /dev/ttyACM0
```

That has to be a native Linux box. Inside WSL the keyboard would land in WSL, not on the Windows desktop with the slides. I guess there's a PS/2 driver for Windows also, but that's for you to experiment.

What goes on the serial line:

| Key   | Make (press) | Break (release) |
|-------|--------------|-----------------|
| Right | `E0 74`      | `E0 F0 74`      |
| Left  | `E0 6B`      | `E0 F0 6B`      |

1200 baud, 8N1. The `E0` prefix is there because both arrows are "extended" keys in PS/2 scan code set 2. One tap is make plus break, five bytes. No debug text ever goes over the UART. That all goes through the probe (RTT).

One tilt, one key. A key fires when the prediction changes to next or prev with at least 75 percent confidence (`CONFIDENT` in main.rs). Holding the tilt does not repeat it. The trigger re-arms when some other gesture is seen with the same confidence, which in practice means bringing the board back to rest. Rest and shake only show on the display.

## The debug output

All of this is RTT over the probe, shown by `cargo run --release`. The buffer is 48 KB, enough for the whole data dump, and it never blocks. With no debugger attached the board runs exactly the same and the lines just go nowhere.

| Line                                                   | When                                            |
|--------------------------------------------------------|-------------------------------------------------|
| `BOOT accelerometer id 0x33`                           | once (0x33 is the LSM303AGR's WHO_AM_I)         |
| `RECORD rest: A stores the last 640 ms, B moves on`    | start of each gesture's recording               |
| `REC next 3/12`                                        | a window saved, with the running count          |
| `TRAIN 48 windows, 200 epochs`                         | B on the fourth gesture                         |
| `E0010 loss 0.123456`                                  | mean loss, every 10 epochs and at epoch 199     |
| `TRAINED 48/48 right on the training windows, 1234 ms` | end of training, timed by a hardware timer      |
| `W 7 next 31 -12 980 215`                              | the data dump, see below                        |
| `WORK`                                                 | entering work mode                              |
| `P next 0.97`                                          | a prediction and its probability, every 200 ms  |
| `KEY Right`                                            | a key was sent                                  |

The numbers above only show the format. The `W` lines are the evidence. After `TRAINED`, every saved window is printed one sample per line: `W <idx> <gesture> <t> <x> <y> <z>`. `idx` is 0 to 47, `t` is 0 to 31 inside the window, and x y z are raw readings in milli-g (1000 is gravity). Keep the log. The research repo has a tool that grades a session from these lines.

`HOWITWORKS.md` is one page on the machine learning part.
