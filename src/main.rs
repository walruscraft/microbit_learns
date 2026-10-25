//! A micro:bit v2 that learns your gestures on itself and then drives your
//! slides. Three stages, top to bottom: record, train, work.
//!
//! Debug output goes over the probe (RTT). The serial port carries only
//! key presses, see keys.rs.
#![no_std]
#![no_main]

mod accel;
mod gesture;
mod keys;
mod leds;
mod mlp;

use cortex_m_rt::entry;
use embedded_hal::delay::DelayNs;
use embedded_hal::digital::InputPin;
use heapless::Vec;
use microbit::board::Board;
use microbit::hal::clocks::Clocks;
use microbit::hal::gpio::{Floating, Input, Pin};
use microbit::hal::timer::Timer;
use microbit::hal::twim::{self, Twim};
use microbit::pac::{TIMER0, TIMER2};
use panic_rtt_target as _;
use rtt_target::{rprintln, rtt_init_print, ChannelMode};

use accel::{Accelerometer, Window, WIN};
use gesture::Gesture;
use keys::Keyboard;
use mlp::Mlp;

/// Windows recorded per gesture.
const PER_CLASS: usize = 12;
/// Passes over the data during training.
const EPOCHS: usize = 200;
/// The main loop runs at the accelerometer's 50 Hz.
const TICK_MS: u32 = 20;
/// Ten ticks, so a prediction every 200 ms in work mode.
const TICKS_PER_PREDICTION: usize = 10;
/// Below this probability a prediction changes nothing: no icon, no key.
const CONFIDENT: f32 = 0.75;

/// One recorded window and what gesture it was.
struct Labelled {
    window: Window,
    gesture: Gesture,
}

const MAX_WINDOWS: usize = PER_CLASS * Gesture::ALL.len();
type Dataset = Vec<Labelled, MAX_WINDOWS>;

/// A button that reports a press once, on the tick it goes down.
struct Button {
    pin: Pin<Input<Floating>>,
    was_down: bool,
}

impl Button {
    /// A button already held at boot is not a press.
    fn new(mut pin: Pin<Input<Floating>>) -> Self {
        let Ok(was_down) = pin.is_low();
        Self { pin, was_down }
    }

    /// Polling every 20 ms is all the debouncing a thumb needs.
    fn pressed(&mut self) -> bool {
        let Ok(down) = self.pin.is_low();
        let pressed = down && !self.was_down;
        self.was_down = down;
        pressed
    }
}

struct Pressed {
    a: bool,
    b: bool,
}

/// What every stage polls: the sensor, the clock, the buttons.
struct Io {
    sensor: Accelerometer,
    timer: Timer<TIMER0>,
    a: Button,
    b: Button,
    window: Window,
}

impl Io {
    /// One 20 ms step: push the newest sample, read the buttons.
    fn tick(&mut self) -> Pressed {
        self.timer.delay_ms(TICK_MS);
        // A failed read keeps the window as it was. The next one will fix it.
        if let Some(sample) = accel::read(&mut self.sensor) {
            self.window.push(sample);
        }
        Pressed {
            a: self.a.pressed(),
            b: self.b.pressed(),
        }
    }
}

#[entry]
fn main() -> ! {
    // Big enough for the whole data dump (about 41 KB) and never blocking,
    // so the board runs the same with no debugger attached.
    rtt_init_print!(ChannelMode::NoBlockSkip, 49152);
    let board = Board::take().expect("main takes the board once");
    // Training time is the headline number of the demo, so the clock is the
    // crystal and not the internal RC oscillator (which runs a few percent
    // fast).
    Clocks::new(board.CLOCK).enable_ext_hfosc();

    let mut timer = Timer::new(board.TIMER0);
    leds::init(board.TIMER1, board.display_pins);
    let i2c = Twim::new(
        board.TWIM0,
        board.i2c_internal.into(),
        twim::Frequency::K400,
    );
    let mut sensor = accel::setup(i2c, &mut timer);
    rprintln!("BOOT accelerometer id {:#04x}", accel::id(&mut sensor));

    let mut io = Io {
        sensor,
        timer,
        a: Button::new(board.buttons.button_a.degrade()),
        b: Button::new(board.buttons.button_b.degrade()),
        window: Window::new(),
    };
    let mut keyboard = Keyboard::new(board.UARTE0, board.uart);
    // The network is 9.7 KB, too big for the stack. singleton! puts it in a
    // static, writes ZERO into it, and hands out the one &mut.
    let model = cortex_m::singleton!(: Mlp = Mlp::ZERO).expect("main runs once");

    let data = record(&mut io);
    train(&data, model, Timer::new(board.TIMER2));
    dump(&data);
    work(&mut io, model, &mut keyboard)
}

/// Stage one. For each gesture in turn: A stores the last 640 ms, B moves on.
fn record(io: &mut Io) -> Dataset {
    let mut data = Dataset::new();
    for gesture in Gesture::ALL {
        leds::show(gesture.icon());
        rprintln!(
            "RECORD {}: A stores the last 640 ms, B moves on",
            gesture.name()
        );
        let mut count = 0;
        loop {
            let pressed = io.tick();
            if pressed.a && count < PER_CLASS {
                let example = Labelled {
                    window: io.window,
                    gesture,
                };
                if data.push(example).is_ok() {
                    count += 1;
                    leds::show_count(count);
                    rprintln!("REC {} {}/{}", gesture.name(), count, PER_CLASS);
                }
            }
            if pressed.b && count > 0 {
                break;
            }
        }
    }
    data
}

/// Stage two. 200 passes over the data, one small weight update per window.
/// In epoch e every window is first shifted in time by e samples, so the
/// network sees each tilt at every offset and learns the tilt, not where in
/// the window it landed.
fn train(data: &Dataset, model: &mut Mlp, mut stopwatch: Timer<TIMER2>) {
    rprintln!("TRAIN {} windows, {} epochs", data.len(), EPOCHS);
    leds::show_progress(0, EPOCHS);
    stopwatch.start(u32::MAX);
    model.init();
    for epoch in 0..EPOCHS {
        let mut loss_sum = 0.0;
        for example in data {
            let x = example.window.shifted(epoch % WIN).features();
            loss_sum += model.train_step(&x, example.gesture);
        }
        if epoch % 10 == 0 || epoch + 1 == EPOCHS {
            rprintln!("E{:04} loss {:.6}", epoch, loss_sum / data.len() as f32);
        }
        leds::show_progress(epoch + 1, EPOCHS);
    }
    let ms = stopwatch.read() / 1000;
    let right = data
        .iter()
        .filter(|e| model.predict(&e.window.features()).0 == e.gesture)
        .count();
    rprintln!(
        "TRAINED {}/{} right on the training windows, {} ms",
        right,
        data.len(),
        ms
    );
}

/// The evidence: every recorded window, one sample per line. It all fits in
/// the RTT buffer, so this is quick.
fn dump(data: &Dataset) {
    for (idx, example) in data.iter().enumerate() {
        for (t, [x, y, z]) in example.window.samples().iter().enumerate() {
            rprintln!(
                "W {} {} {} {} {} {}",
                idx,
                example.gesture.name(),
                t,
                x,
                y,
                z
            );
        }
    }
}

/// Stage three, forever. A prediction every 200 ms. Below CONFIDENT nothing
/// happens. A confident next or prev shows its icon and taps its key once.
/// It re-arms when some other gesture is seen with confidence.
fn work(io: &mut Io, model: &Mlp, keyboard: &mut Keyboard) -> ! {
    rprintln!("WORK");
    // The window is stale (it still holds the B press from before training),
    // so refill it.
    for _ in 0..WIN {
        io.tick();
    }
    let mut last = Gesture::Rest;
    loop {
        for _ in 0..TICKS_PER_PREDICTION {
            io.tick();
        }
        let (gesture, prob) = model.predict(&io.window.features());
        rprintln!("P {} {:.2}", gesture.name(), prob);
        if prob < CONFIDENT {
            continue;
        }
        leds::show(gesture.icon());
        if gesture != last {
            if let Some(key) = gesture.key() {
                keyboard.tap(key);
                rprintln!("KEY {:?}", key);
            }
            last = gesture;
        }
    }
}
