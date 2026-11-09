//! Keeping rows of control points continuous and smooth along a chain's breaks.

use brepkit_math::vec::Vec3;

use crate::OperationsError;

/// A jump bigger than this at a break is a property of the data, not rounding,
/// and gets a narrow band of its own to close over rather than spreading
/// across a whole piece.
const WIDE_JUMP: f64 = 1e-7;

/// How much of the shorter piece beside a wide jump its band may take.
const BAND: f64 = 0.1;

/// Breaks closer than this are one break.
const BREAK_MERGE: f64 = 1e-7;

/// The tolerance a row of control points is fitted to: small enough against
/// `smallest`, the distance across the surface, for the seam's second
/// derivative to hold, and no smaller than rounding allows at `largest`, its
/// largest coordinate.
pub(super) fn row_tolerance(smallest: f64, largest: f64) -> f64 {
    (1e-10 * smallest.min(1.0)).max(1e-12 * largest)
}

/// Sort `breaks` and merge those that nearly coincide, keeping the ends.
pub(super) fn dedup_breaks(breaks: &mut Vec<f64>) {
    for b in breaks.iter_mut() {
        *b = b.clamp(0.0, 1.0);
    }
    breaks.sort_by(f64::total_cmp);
    breaks.dedup_by(|b, a| *b - *a < BREAK_MERGE);
    let last = breaks.len() - 1;
    if 1.0 - breaks[last] < BREAK_MERGE {
        breaks[last] = 1.0;
    } else {
        breaks.push(1.0);
    }
}

/// The piece that holds `s`, a break reading as the piece it starts.
pub(super) fn piece_of(breaks: &[f64], s: f64) -> usize {
    breaks
        .partition_point(|b| *b <= s)
        .saturating_sub(1)
        .min(breaks.len() - 2)
}

/// `6x^5 - 15x^4 + 10x^3`: 0 to 1, flat to second order at both ends.
pub(super) fn smoother(x: f64) -> f64 {
    x * x * x * (x * (6.0 * x - 15.0) + 10.0)
}

/// What each piece's rows need added to meet at its breaks.
pub(super) struct Ramps {
    /// `(jump at start of piece, jump at end of piece)`, each half the gap.
    ends: Vec<(Vec<Vec3>, Vec<Vec3>)>,
    wide: Vec<usize>,
}

impl Ramps {
    /// Read the rows from both sides of each break of `breaks`.
    ///
    /// `rows(piece, lo, hi, s)` gives the rows at `s` as read from the piece.
    pub fn measure<F>(
        breaks: &[f64],
        closed: bool,
        tolerance: f64,
        mut rows: F,
    ) -> Result<Self, OperationsError>
    where
        F: FnMut(usize, f64, f64, f64) -> Result<Vec<Vec3>, OperationsError>,
    {
        let pieces = breaks.len() - 1;
        let probe = rows(0, breaks[0], breaks[1], breaks[0])?;
        let zero = vec![Vec3::new(0.0, 0.0, 0.0); probe.len()];
        let mut ends = vec![(zero.clone(), zero); pieces];
        let mut wide = Vec::new();
        let joins = if closed { pieces } else { pieces - 1 };
        for join in 0..joins {
            let (left, right) = (join, (join + 1) % pieces);
            let (l_lo, l_hi) = (breaks[left], breaks[left + 1]);
            let (r_lo, r_hi) = (breaks[right], breaks[right + 1]);
            let from_left = rows(left, l_lo, l_hi, l_hi)?;
            let from_right = rows(right, r_lo, r_hi, r_lo)?;
            let mut biggest = 0.0_f64;
            for (k, (a, b)) in from_left.iter().zip(&from_right).enumerate() {
                let half = (*b - *a) * 0.5;
                biggest = biggest.max(half.length());
                ends[left].1[k] = half;
                ends[right].0[k] = -half;
            }
            if biggest > WIDE_JUMP.max(tolerance) {
                wide.push(join);
            }
        }
        Ok(Self { ends, wide })
    }

    /// Whether some break's jump asks for a band of its own.
    pub fn needs_bands(&self) -> bool {
        !self.wide.is_empty()
    }

    /// The breaks that bound each wide jump's band.
    pub fn band_breaks(&self, breaks: &[f64]) -> Vec<f64> {
        let pieces = breaks.len() - 1;
        let mut found = Vec::new();
        for &join in &self.wide {
            let (left, right) = (join, (join + 1) % pieces);
            let width =
                BAND * (breaks[left + 1] - breaks[left]).min(breaks[right + 1] - breaks[right]);
            found.push(breaks[left + 1] - width);
            found.push(breaks[right] + width);
        }
        found
    }

    /// Add this piece's share of the jumps at its ends to `rows` at `s`.
    pub fn apply(&self, piece: usize, lo: f64, hi: f64, s: f64, rows: &mut [Vec3]) {
        let along = ((s - lo) / (hi - lo)).clamp(0.0, 1.0);
        let (at_start, at_end) = &self.ends[piece];
        let (from_start, from_end) = (smoother(1.0 - along), smoother(along));
        for (k, row) in rows.iter_mut().enumerate() {
            *row = *row + at_start[k] * from_start + at_end[k] * from_end;
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    #[test]
    fn close_breaks_merge_and_the_ends_stay() {
        let mut breaks = vec![0.5, 0.0, 1.0, 0.5 + 1e-9, 0.25, 1.0 - 1e-9];
        dedup_breaks(&mut breaks);
        assert_eq!(breaks, vec![0.0, 0.25, 0.5, 1.0]);
    }

    #[test]
    fn a_break_reads_as_the_piece_it_starts() {
        let breaks = [0.0, 0.25, 0.5, 1.0];
        assert_eq!(piece_of(&breaks, 0.0), 0);
        assert_eq!(piece_of(&breaks, 0.25), 1);
        assert_eq!(piece_of(&breaks, 0.3), 1);
        assert_eq!(piece_of(&breaks, 1.0), 2);
    }

    #[test]
    fn rows_meet_at_their_break() {
        let breaks = [0.0, 0.5, 1.0];
        let ramps = Ramps::measure(&breaks, false, 1e-10, |piece, _, _, _| {
            Ok(vec![Vec3::new(
                if piece == 0 { 1.0 } else { 1.5 },
                0.0,
                0.0,
            )])
        })
        .unwrap();
        assert!(ramps.needs_bands());
        let mut left = vec![Vec3::new(1.0, 0.0, 0.0)];
        ramps.apply(0, 0.0, 0.5, 0.5, &mut left);
        let mut right = vec![Vec3::new(1.5, 0.0, 0.0)];
        ramps.apply(1, 0.5, 1.0, 0.5, &mut right);
        assert!((left[0] - right[0]).length() < 1e-12);
    }
}
