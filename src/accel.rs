//! The accelerometer, and the window of its samples that the network looks at.

use embedded_hal::delay::DelayNs;
use lsm303agr::interface::I2cInterface;
use lsm303agr::mode::MagOneShot;
use lsm303agr::{AccelMode, AccelOutputDataRate, Lsm303agr};
use microbit::hal::twim::Twim;
use microbit::pac::TWIM0;

/// Samples per window: 640 ms at 50 Hz, enough for one tilt and back.
pub const WIN: usize = 32;
pub const AXES: usize = 3;
/// What the network sees: every sample of every axis, flattened.
pub const FEATS: usize = WIN * AXES;

/// One reading, x y z, in milli-g (1000 is gravity).
pub type Sample = [i16; AXES];
pub type Features = [f32; FEATS];

/// The LSM303AGR on the board's internal I2C bus, accelerometer half only.
pub type Accelerometer = Lsm303agr<I2cInterface<Twim<TWIM0>>, MagOneShot>;

/// Wakes the chip up at 50 Hz, 12-bit resolution, +-2 g.
pub fn setup(i2c: Twim<TWIM0>, delay: &mut impl DelayNs) -> Accelerometer {
    let mut sensor = Lsm303agr::new_with_i2c(i2c);
    sensor.init().expect("accelerometer init");
    sensor
        .set_accel_mode_and_odr(delay, AccelMode::HighResolution, AccelOutputDataRate::Hz50)
        .expect("accelerometer mode");
    sensor
}

/// The chip's WHO_AM_I byte. 0x33 means it is the chip we expect.
pub fn id(sensor: &mut Accelerometer) -> u8 {
    sensor.accelerometer_id().expect("accelerometer id").raw()
}

/// None when the I2C bus had a bad moment.
pub fn read(sensor: &mut Accelerometer) -> Option<Sample> {
    let (x, y, z) = sensor.acceleration().ok()?.xyz_mg();
    // +-2 g is +-2000, so this never actually fails.
    let mg = |v: i32| i16::try_from(v).expect("milli-g fits in an i16");
    Some([mg(x), mg(y), mg(z)])
}

/// The last 640 ms of movement, oldest sample first.
#[derive(Clone, Copy)]
pub struct Window {
    samples: [Sample; WIN],
}

impl Window {
    pub const fn new() -> Self {
        Self {
            samples: [[0; AXES]; WIN],
        }
    }

    /// Drops the oldest sample and appends the newest.
    pub fn push(&mut self, sample: Sample) {
        self.samples.rotate_left(1);
        self.samples[WIN - 1] = sample;
    }

    pub fn samples(&self) -> &[Sample; WIN] {
        &self.samples
    }

    /// The window rotated by `by` samples. Your thumb catches the tilt
    /// somewhere in the 640 ms, and later the sliding window does the same,
    /// so the network has to know the tilt in every position. Rotating gives
    /// all 32 positions for free. When the cut lands inside the tilt its two
    /// halves swap (you can see that as small bumps in the loss), but the
    /// sign and size of the movement survive, and that is what tells the
    /// four gestures apart.
    pub fn shifted(&self, by: usize) -> Window {
        let mut copy = *self;
        copy.samples.rotate_left(by % WIN);
        copy
    }

    /// What the network eats. Each axis gets its mean removed (that takes
    /// out gravity and the angle you hold the board at) and is scaled from
    /// milli-g to g. The scale is fixed on purpose. The size of the movement
    /// matters, and normalizing each window would make a still board look
    /// like a tiny shake.
    pub fn features(&self) -> Features {
        let mean: [f32; AXES] = core::array::from_fn(|axis| {
            let sum: f32 = self.samples.iter().map(|s| f32::from(s[axis])).sum();
            sum / WIN as f32
        });
        let mut out = [0.0; FEATS];
        for (t, sample) in self.samples.iter().enumerate() {
            for axis in 0..AXES {
                out[t * AXES + axis] = (f32::from(sample[axis]) - mean[axis]) / 1000.0;
            }
        }
        out
    }
}
