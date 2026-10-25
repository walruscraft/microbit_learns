//! The board as a keyboard. It sends PS/2 scan codes down the serial line
//! at 1200 baud, and the stock Linux driver turns them into key presses:
//!
//!     sudo inputattach --ps2serkbd /dev/ttyACM0
//!
//! Same trick as the 2024 micro:bit meetup, two keys instead of a mouse.

use microbit::board::UartPins;
use microbit::hal::uarte::{Baudrate, Parity, Uarte};
use microbit::pac::UARTE0;

use crate::gesture::Gesture;

#[derive(Clone, Copy, Debug)]
pub enum Key {
    Right,
    Left,
}

impl Key {
    /// PS/2 scan code set 2. Both arrows are "extended" keys: E0 then this.
    fn code(self) -> u8 {
        match self {
            Key::Right => 0x74,
            Key::Left => 0x6B,
        }
    }
}

impl Gesture {
    pub fn key(self) -> Option<Key> {
        match self {
            Gesture::Next => Some(Key::Right),
            Gesture::Prev => Some(Key::Left),
            Gesture::Rest | Gesture::Shake => None,
        }
    }
}

pub struct Keyboard {
    uart: Uarte<UARTE0>,
}

impl Keyboard {
    pub fn new(uarte: UARTE0, pins: UartPins) -> Self {
        let uart = Uarte::new(uarte, pins.into(), Parity::EXCLUDED, Baudrate::BAUD1200);
        Self { uart }
    }

    /// Press and release: make code E0 xx, break code E0 F0 xx.
    pub fn tap(&mut self, key: Key) {
        let code = key.code();
        // On the stack, because the UART's DMA cannot read from flash.
        let bytes = [0xE0, code, 0xE0, 0xF0, code];
        self.uart.write(&bytes).expect("uart write");
    }
}
