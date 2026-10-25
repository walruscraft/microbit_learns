# How it works

One page on the machine learning in this program, for Rust programmers.
There is no framework. The whole network is in `src/mlp.rs` and its input
is made in `src/accel.rs`. The function names below tell you where to
look.

## The window

The accelerometer runs at 50 Hz and reports x, y, z in milli-g (1000 is
gravity, and the chip is set to +-2 g). The program keeps a `Window` of
the last 32 samples, which is 640 ms, oldest first. Every 20 ms the
oldest goes out and the newest comes in (`Window::push`). Pressing A
while recording saves the window as it is right then, with its label.
That is why you press A as the tilt ends: the tilt is then inside the
last 640 ms.

A window is 32 x 3 = 96 `i16` values, 192 bytes. The board stores 48 of
them (12 per gesture) in a fixed-size `heapless::Vec`. Nothing allocates.

## The features

`Window::features` turns a window into the 96 `f32` numbers the network
sees. Two things happen to each axis:

1. Subtract the mean of the axis over the window. That removes gravity
   and whatever angle you hold the board at, so "rest" is a flat line
   near zero and a tilt is the change.
2. Divide by 1000, so milli-g becomes g.

That's all. The scale is fixed on purpose. The common reflex is to
normalize each window to unit size, but that would turn a still board
into something that looks like a tiny shake. Here the size of the
movement is information, so it stays.

## The network

`Mlp` is short for multilayer perceptron, the oldest and plainest kind of
neural network. (Search the web for MLP and you get My Little Pony first.
Ours is the other one. Fewer ponies, same amount of magic.) It is 96
inputs, 24 hidden units with ReLU, 4 outputs with softmax (one per
gesture, in the order of `Gesture::ALL`).

| Layer    | Weights | Biases | Parameters |
|----------|---------|--------|------------|
| 96 -> 24 | 2304    | 24     | 2328       |
| 24 -> 4  | 96      | 4      | 100        |
| total    |         |        | 2428       |

2428 `f32` parameters is about 9.7 KB, as four plain arrays in a struct.
The forward pass (`Mlp::forward`) is two matrix-vector products:
`h = relu(W1 x + b1)`, then `p = softmax(W2 h + b2)`. Softmax subtracts
the largest logit before `expf`, so it cannot overflow.

Starting weights are uniform in `[-lim, lim)` with
`lim = sqrt(6 / fan_in)`, the usual choice. Biases start at zero. The
weights are not random, though. An init only needs every unit to start
different from the others, so the n-th weight is the fractional part of
n times the golden ratio (`spread` in mlp.rs). It is done in 32-bit
integers: multiply n by 0x9E3779B9, which is (phi - 1) times 2^32, and
let it wrap. The wrap is the fractional part. Those fractions spread
evenly over [0, 1) and never repeat (sunflower seeds fill a head the same
way), and it needs no library at all. So the same tilts always give the
same network, bit for bit.

## The loss and its gradient

The loss is cross-entropy, `-ln(p[target])`, with `p` clamped at `1e-7`
so the log never hits infinity. Softmax plus cross-entropy has the
gradient everyone remembers: `p - y`, the network's probabilities minus
one at the right answer. From there `Mlp::train_step` walks the chain
rule back by hand:

1. `d_logits = p - y`.
2. `d_hidden = W2^T d_logits`, computed before W2 changes.
3. Update W2 and b2 by `lr * d_logits * h`.
4. Where a hidden unit was off (`h_pre <= 0`), ReLU let nothing through
   and that row of W1 gets no update. Elsewhere, update W1 and b1 by
   `lr * d_hidden * x`.

`train_step` returns the loss before the step. The `E` lines in the log
are that, averaged over the 48 windows.

## Training

`train` in main.rs:

- 200 epochs. Each epoch, every window (in recording order) gets one
  `train_step`. Plain per-sample gradient descent, learning rate 0.02
  (`LR`).
- In epoch e every window is first rotated in time by e samples
  (`Window::shifted`, wrapping at 32), and the features come from the
  rotated copy. Over 200 epochs each window is seen at all 32 offsets,
  about six times each, so the network learns the tilt and not where in
  the 640 ms it happened to land.
- Nothing is random, so two runs on the same windows print the same `E`
  lines.
- No shuffling. The order is rest x 12, next x 12, prev x 12, shake x 12,
  every epoch.

The spiral on the display advances one pixel every 8 epochs from a timer
interrupt, so the main loop does nothing but train.

## Prediction

`Mlp::predict` runs the forward pass and returns the most likely gesture
(argmax) and its probability. In work mode that runs every 200 ms on the
live window. Below 0.75 the board changes nothing, no icon and no key.
That keeps the first quarter second of a shake from flashing an arrow.

## Why 12 windows per class is enough

Twelve examples per class would get you laughed out of most ML rooms. It
works here because the four motions are very different (still, right,
left, side to side), the window is short, and the network is tiny. 2428
parameters cannot memorize much, and the time shift turns each window
into up to 32 different-looking ones. The real sessions in the research
repo were recorded this way, and session 2 got 46 of 48 held-out windows
right. Twelve was enough for a human.

## What is deliberately not done

- No held-out set on the board. The `TRAINED` line is accuracy on the
  training windows, which is grading your own homework. The honest number
  comes from replaying the `W` lines on a laptop.
- No quantization. Everything is `f32`. The nRF52833 has an FPU, and
  9.7 KB fits fine in 128 KB of RAM.
- No shuffling. Textbooks say to shuffle. The time shift changes what
  each epoch sees anyway, and the loop stays three lines.
- No early stopping. Always 200 epochs, so every run does the same work
  and the `E` lines of two runs line up.
