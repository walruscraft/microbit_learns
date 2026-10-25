//! The four things the board learns to tell apart.

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Gesture {
    /// Holding still, at whatever angle.
    Rest,
    /// Tilt to the right: next slide.
    Next,
    /// Tilt to the left: previous slide.
    Prev,
    /// Quick tilting side to side: just for show.
    Shake,
}

impl Gesture {
    /// Recording order, and the order of the network's four outputs.
    pub const ALL: [Gesture; 4] = [Gesture::Rest, Gesture::Next, Gesture::Prev, Gesture::Shake];

    /// Position in `ALL`, which is also the index of its output neuron.
    pub fn index(self) -> usize {
        self as usize
    }

    pub fn name(self) -> &'static str {
        match self {
            Gesture::Rest => "rest",
            Gesture::Next => "next",
            Gesture::Prev => "prev",
            Gesture::Shake => "shake",
        }
    }
}
