//! The network, an MLP. That is a multilayer perceptron, and if you got
//! here from a web search, the ponies are elsewhere.
//! 96 inputs, 24 hidden units with ReLU, 4 softmax outputs.
//! Plain arrays, about 9.7 KB, no heap. The few lines of math every neural
//! network is made of are written out here, so you do not have to trust a
//! framework (or me).

use crate::accel::{Features, FEATS};
use crate::gesture::Gesture;

pub const HID: usize = 24;
pub const OUT: usize = Gesture::ALL.len();
/// Learning rate: how far one sample pushes the weights.
pub const LR: f32 = 0.02;

pub struct Mlp {
    w1: [[f32; FEATS]; HID],
    b1: [f32; HID],
    w2: [[f32; HID]; OUT],
    b2: [f32; OUT],
}

/// What the forward pass leaves behind. Backprop needs it again.
struct Activations {
    /// Hidden layer before ReLU. Its sign is the ReLU gate.
    h_pre: [f32; HID],
    h: [f32; HID],
    probs: [f32; OUT],
}

impl Mlp {
    /// All zeros. A const, so it takes no memory by itself. The one real
    /// Mlp is the static that singleton! makes in main.
    pub const ZERO: Mlp = Mlp {
        w1: [[0.0; FEATS]; HID],
        b1: [0.0; HID],
        w2: [[0.0; HID]; OUT],
        b2: [0.0; OUT],
    };

    /// Small, well-spread starting weights and zero biases. An init needs
    /// every unit to start different and nothing to start large. It does not
    /// need randomness, so there is none, and the same tilts always give the
    /// same network. The range is the usual sqrt(6 / fan_in).
    pub fn init(&mut self) {
        // Back to zero first, so init is a full reset and not just new weights.
        *self = Self::ZERO;
        let lim1 = libm::sqrtf(6.0 / FEATS as f32);
        let lim2 = libm::sqrtf(6.0 / HID as f32);
        let mut n = 0;
        for w in self.w1.iter_mut().flatten() {
            n += 1;
            *w = spread(n, lim1);
        }
        for w in self.w2.iter_mut().flatten() {
            n += 1;
            *w = spread(n, lim2);
        }
    }

    fn forward(&self, x: &Features) -> Activations {
        let mut act = Activations {
            h_pre: [0.0; HID],
            h: [0.0; HID],
            probs: [0.0; OUT],
        };
        for (h, (row, bias)) in self.w1.iter().zip(&self.b1).enumerate() {
            let pre = bias + dot(row, x);
            act.h_pre[h] = pre;
            act.h[h] = pre.max(0.0);
        }
        let logits: [f32; OUT] = core::array::from_fn(|o| self.b2[o] + dot(&self.w2[o], &act.h));
        act.probs = softmax(&logits);
        act
    }

    /// The most likely gesture and how sure the network is.
    pub fn predict(&self, x: &Features) -> (Gesture, f32) {
        let probs = self.forward(x).probs;
        let (best, &prob) = probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .expect("four outputs is at least one");
        (Gesture::ALL[best], prob)
    }

    /// One step of gradient descent on one labelled window. Returns the loss
    /// before the step, for the log.
    pub fn train_step(&mut self, x: &Features, target: Gesture) -> f32 {
        let act = self.forward(x);
        let target = target.index();
        let loss = -libm::logf(act.probs[target].max(1e-7));

        // Softmax with cross-entropy has the famous gradient p - y: the
        // network's probabilities, minus one at the right answer.
        let mut d_logits = act.probs;
        d_logits[target] -= 1.0;

        // Chain rule back through layer 2, before layer 2 changes.
        let mut d_hidden = [0.0; HID];
        for (row, d) in self.w2.iter().zip(&d_logits) {
            for (dh, w) in d_hidden.iter_mut().zip(row) {
                *dh += d * w;
            }
        }

        for ((row, bias), d) in self.w2.iter_mut().zip(&mut self.b2).zip(&d_logits) {
            *bias -= LR * d;
            for (w, h) in row.iter_mut().zip(&act.h) {
                *w -= LR * d * h;
            }
        }

        // Where the unit was off, ReLU let nothing through, so that row gets
        // no update.
        for (h, (row, bias)) in self.w1.iter_mut().zip(&mut self.b1).enumerate() {
            if act.h_pre[h] <= 0.0 {
                continue;
            }
            let d = d_hidden[h];
            *bias -= LR * d;
            for (w, xi) in row.iter_mut().zip(x) {
                *w -= LR * d * xi;
            }
        }
        loss
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}

/// The n-th starting weight, with no RNG and no library. Multiples of the
/// golden ratio, fractional part only, land evenly over [0, 1) and never
/// repeat (sunflower seeds do the same thing), so the weights come out
/// spread out and all different.
fn spread(n: u32, lim: f32) -> f32 {
    // 1. Multiply by (phi - 1) * 2^32. The 32-bit wrap-around keeps the
    //    fractional part, which is the whole trick.
    let frac_u32 = n.wrapping_mul(0x9E37_79B9);
    // 2. The top 24 bits fit an f32 exactly: [0.0, 1.0).
    let unit = (frac_u32 >> 8) as f32 * (1.0 / 16_777_216.0);
    // 3. Scale to [-lim, +lim).
    lim * (2.0 * unit - 1.0)
}

/// Turns scores into probabilities that sum to one. The max is subtracted
/// first so expf never overflows.
fn softmax(logits: &[f32; OUT]) -> [f32; OUT] {
    let max = logits.iter().copied().fold(f32::MIN, f32::max);
    let exps = logits.map(|l| libm::expf(l - max));
    let sum: f32 = exps.iter().sum();
    exps.map(|e| e / sum)
}
