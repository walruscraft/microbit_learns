//! The 5x5 LED matrix. The board support crate scans it from a timer
//! interrupt, so the rest of the program just says what to show.

use core::cell::RefCell;

use cortex_m::interrupt::Mutex;
use microbit::display::nonblocking::{BitImage, Display};
use microbit::gpio::DisplayPins;
use microbit::pac::{interrupt, Interrupt, NVIC, TIMER1};

use crate::gesture::Gesture;

/// Five rows of five pixels, top row first, 1 is lit.
pub type Icon = [[u8; 5]; 5];

/// The display, shared between the main program and the TIMER1 interrupt.
static DISPLAY: Mutex<RefCell<Option<Display<TIMER1>>>> = Mutex::new(RefCell::new(None));

pub fn init(timer: TIMER1, pins: DisplayPins) {
    let display = Display::new(timer, pins);
    cortex_m::interrupt::free(|cs| DISPLAY.borrow(cs).replace(Some(display)));
    // The driver set up TIMER1, but the core still has to be told to listen.
    // Safe: the handler below is the only other user of DISPLAY, and both
    // sides go through the critical section.
    unsafe { NVIC::unmask(Interrupt::TIMER1) };
}

#[interrupt]
fn TIMER1() {
    cortex_m::interrupt::free(|cs| {
        if let Some(display) = DISPLAY.borrow(cs).borrow_mut().as_mut() {
            display.handle_display_event();
        }
    });
}

pub fn show(icon: &Icon) {
    cortex_m::interrupt::free(|cs| {
        if let Some(display) = DISPLAY.borrow(cs).borrow_mut().as_mut() {
            display.show(&BitImage::new(icon));
        }
    });
}

/// `n` dots, filling the grid row by row from the top left.
pub fn show_count(n: usize) {
    let mut icon: Icon = [[0; 5]; 5];
    icon.iter_mut().flatten().take(n).for_each(|p| *p = 1);
    show(&icon);
}

/// `done` of `total` as a spiral from the corner to the centre, rounded up
/// so the very first step already lights a pixel. The centre comes last,
/// when the thinking is over.
pub fn show_progress(done: usize, total: usize) {
    let lit = (done * SPIRAL.len()).div_ceil(total);
    let mut icon: Icon = [[0; 5]; 5];
    for &(row, col) in SPIRAL.iter().take(lit) {
        icon[row][col] = 1;
    }
    show(&icon);
}

/// (row, col) of every pixel, clockwise from the top left corner inwards.
#[rustfmt::skip]
const SPIRAL: [(usize, usize); 25] = [
    (0, 0), (0, 1), (0, 2), (0, 3), (0, 4),
    (1, 4), (2, 4), (3, 4), (4, 4),
    (4, 3), (4, 2), (4, 1), (4, 0),
    (3, 0), (2, 0), (1, 0),
    (1, 1), (1, 2), (1, 3),
    (2, 3), (3, 3),
    (3, 2), (3, 1),
    (2, 1),
    (2, 2),
];

impl Gesture {
    pub fn icon(self) -> &'static Icon {
        match self {
            Gesture::Rest => &REST,
            Gesture::Next => &NEXT,
            Gesture::Prev => &PREV,
            Gesture::Shake => &SHAKE,
        }
    }
}

const REST: Icon = [
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
];
const NEXT: Icon = [
    [0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0],
    [1, 1, 1, 1, 1],
    [0, 0, 0, 1, 0],
    [0, 0, 1, 0, 0],
];
const PREV: Icon = [
    [0, 0, 1, 0, 0],
    [0, 1, 0, 0, 0],
    [1, 1, 1, 1, 1],
    [0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0],
];
const SHAKE: Icon = [
    [1, 0, 1, 0, 1],
    [0, 1, 0, 1, 0],
    [1, 0, 1, 0, 1],
    [0, 1, 0, 1, 0],
    [1, 0, 1, 0, 1],
];
