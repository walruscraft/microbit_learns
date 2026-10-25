# microbit_learns

A BBC micro:bit v2 that learns your hand gestures, trains on itself (no
PC involved), and then flips your slides. You tilt the board a dozen
times per gesture, it trains a small neural network on that, and from
then on a tilt right sends Right arrow and a tilt left sends Left arrow
to the laptop with the slides. The weights are born on the nRF52833,
from your hand.

This is the demo for a 60 minute talk at a Rust meetup. It is written for
Rust programmers, not ML people, so it is six short files in no_std Rust
that you can read top to bottom. No framework, no magic.

## Hardware

A micro:bit v2 and a USB cable. That's it. The accelerometer (an
LSM303AGR) is already on the board.

## Build and flash

The toolchain comes from `rust-toolchain.toml` (stable, plus the
`thumbv7em-none-eabihf` target). The flasher is probe-rs:

```bash
cargo install probe-rs-tools
cargo run --release
```

`cargo run` builds, flashes over USB and stays attached to show the debug
output. That output is plain text over RTT (through the probe, not the
serial port). A panic shows up there too, with file and line.

WSL note: the board belongs to Windows, so hand it to WSL first with
usbipd-win. It shows up as VID:PID 0d28:0204:

```bash
usbipd list                          # find the busid of 0d28:0204
usbipd attach --wsl --busid <busid>
```

The busid changes with the USB port, so look it up each time. After that
WSL sees a CMSIS-DAP probe and a serial port (`/dev/ttyACM0`). For
driving slides the serial port has to be on a real Linux machine though,
see below.

## The demo script

Reset the board (or flash it) and it starts recording. The four gestures
go in a fixed order, and the display shows the icon of the one it wants:

| Order | Gesture | What you do                   | Icon         |
|-------|---------|-------------------------------|--------------|
| 1     | rest    | hold still, at any angle      | one dot      |
| 2     | next    | tilt right                    | right arrow  |
| 3     | prev    | tilt left                     | left arrow   |
| 4     | shake   | quick tilts side to side      | checkerboard |

The board reads the accelerometer every 20 ms and always keeps the last
640 ms (32 samples). Button A saves that window as one example of the
current gesture. So: do the tilt, press A as it ends. Every press adds a
dot to the display so you can count. Twelve per gesture, after that A
does nothing. Button B moves on to the next gesture (it is ignored while
the current one has zero examples). B on the fourth gesture starts
training.

Training is 200 passes over the 48 windows. The display draws a spiral
from the corner to the centre, one pixel per 8 passes. When the centre
lights up, the thinking is over and the board is in work mode, for good.
It predicts every 200 ms and, when at least 75 percent sure, shows the
icon of what it thinks you are doing. The buttons do nothing now (press
them anyway, everybody does). Reset starts over with fresh weights.

The LED language, in one table:

| Display                                    | Meaning                                         |
|--------------------------------------------|-------------------------------------------------|
| dot, right arrow, left arrow, checkerboard | rest, next, prev, shake. While recording: the one it wants. While working: the one it sees |
| n dots, from the top left                  | n examples saved for this gesture               |
| spiral, growing                            | training progress (the centre pixel is last)    |

## Driving the slides

The board pretends to be a keyboard. Same trick as the 2024 meetup, where
example 5 of <https://github.com/walruscraft/rust-examples-microbit-2024>
sent PS/2 mouse codes over the UART. This time it is two keys instead of
a mouse. On the Linux machine with the slides:

```bash
sudo inputattach --ps2serkbd /dev/ttyACM0
```

That has to be a native Linux box. Inside WSL the keyboard would land in
WSL, not on the Windows desktop with the slides.

What goes on the wire, and nothing else, ever:

| Key   | Make (press) | Break (release) |
|-------|--------------|-----------------|
| Right | `E0 74`      | `E0 F0 74`      |
| Left  | `E0 6B`      | `E0 F0 6B`      |

1200 baud, 8N1. The `E0` prefix is there because both arrows are
"extended" keys in PS/2 scan code set 2. One tap is make plus break, five
bytes. No debug text ever goes over the UART. That all goes through the
probe.

One tilt, one key. A key fires when the prediction changes to next or
prev with at least 75 percent confidence (`CONFIDENT` in main.rs).
Holding the tilt does not repeat it. The trigger re-arms when some other
gesture is seen with the same confidence, which in practice means
bringing the board back to rest. Rest and shake only show on the display.

## The debug output

All of this is RTT over the probe, shown by `cargo run --release`. The
buffer is 48 KB, enough for the whole data dump, and it never blocks.
With no debugger attached the board runs exactly the same and the lines
just go nowhere.

| Line                                           | When                                            |
|------------------------------------------------|-------------------------------------------------|
| `BOOT accelerometer id 0x33`                   | once (0x33 is the LSM303AGR's WHO_AM_I)        |
| `RECORD rest: A stores the last 640 ms, B moves on` | start of each gesture's recording          |
| `REC next 3/12`                                | a window saved, with the running count          |
| `TRAIN 48 windows, 200 epochs`                 | B on the fourth gesture                         |
| `E0010 loss 0.123456`                          | mean loss, every 10 epochs and at epoch 199     |
| `TRAINED 48/48 right on the training windows, 1234 ms` | end of training, timed by a hardware timer |
| `W 7 next 31 -12 980 215`                      | the data dump, see below                        |
| `WORK`                                         | entering work mode                              |
| `P next 0.97`                                  | a prediction and its probability, every 200 ms  |
| `KEY Right`                                    | a key was sent                                  |

The numbers above only show the format. The `W` lines are the evidence.
After `TRAINED`, every saved window is printed one sample per line:
`W <idx> <gesture> <t> <x> <y> <z>`. `idx` is 0 to 47, `t` is 0 to 31
inside the window, and x y z are raw readings in milli-g (1000 is
gravity). Keep the log. The research repo has a tool that grades a
session from these lines.

## Code map

| File             | What it is                                                        |
|------------------|-------------------------------------------------------------------|
| `src/main.rs`    | boot, the button and tick loop, then the three stages: record, train, work |
| `src/accel.rs`   | the LSM303AGR at 50 Hz, the 32-sample window, and the features it becomes |
| `src/mlp.rs`     | the network: forward, predict, one training step with the gradient spelled out |
| `src/leds.rs`    | the 5x5 matrix: icons, the dot count, the spiral, the display interrupt |
| `src/keys.rs`    | Right and Left as PS/2 scan codes over the UART                   |
| `src/gesture.rs` | the `Gesture` enum: the four classes, their order and names       |

`HOWITWORKS.md` is one page on the machine learning part.

## Numbers

The honest version. On 2026-10-15 this program trained on the real board
three times, 48 windows each, one person tilting. With the code as it is
now: 6235 ms for 200 epochs, 48 of 48 right on its own windows, all 1536
dump lines received, and seven tilts in a row gave seven correct keys.
(The 48 of 48 is on the training data, so it flatters.)

The held-out number comes from the research repo
(<https://github.com/walruscraft/ml_webinar_2026>, private). Two earlier
sessions were recorded with the research firmware using the same recipe:
same 640 ms window, same features, same 96-24-4 network, same training.
Session 2, from 2026-10-14, trained on the board in 9663 ms. A laptop
replay of the board's own data got 46 of 48 held-out windows right. Live
after training, one gesture of each kind: 4 of 4. That is the number to
believe.

## Not in this repo

- Tests.
- Host tools (the log grader and the serial-to-keys bridge).
- The QEMU build.
- The laptop replay of a session. The program is deterministic (fixed
  starting weights, fixed time shifts), so the same windows always give
  the same network. The replay tool just lives with the research
  firmware's log format.

All of that is in the research repo. This is the one the audience reads.
