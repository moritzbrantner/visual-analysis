//! Minimal 3D point, vector and segment types for posture skeletons.
//!
//! Posture keeps its own small `f32` geometry rather than depending on a
//! spatial-reconstruction crate. The serialized shapes (`{x, y, z}` and
//! `{start, end}`) match the former `three-d-processing-core` types.

use std::ops::{Add, Div, Mul, Sub};

use serde::{Deserialize, Serialize};
use video_analysis_core::Result;

use crate::invalid_argument;

/// A 3D direction or displacement.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct Vector3 {
    /// The x component.
    pub x: f32,
    /// The y component.
    pub y: f32,
    /// The z component.
    pub z: f32,
}

impl Vector3 {
    /// The zero vector.
    pub const ZERO: Self = Self::new(0.0, 0.0, 0.0);

    /// Creates a new vector.
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    /// Returns whether every component is finite.
    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }

    /// Returns the dot product.
    pub fn dot(self, rhs: Self) -> f32 {
        self.x.mul_add(rhs.x, self.y.mul_add(rhs.y, self.z * rhs.z))
    }

    /// Returns the Euclidean length.
    pub fn length(self) -> f32 {
        self.dot(self).sqrt()
    }
}

impl Add for Vector3 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

impl Sub for Vector3 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

impl Mul<f32> for Vector3 {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        Self::new(self.x * rhs, self.y * rhs, self.z * rhs)
    }
}

impl Div<f32> for Vector3 {
    type Output = Self;

    fn div(self, rhs: f32) -> Self::Output {
        Self::new(self.x / rhs, self.y / rhs, self.z / rhs)
    }
}

/// A 3D position.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct Point3 {
    /// The x coordinate.
    pub x: f32,
    /// The y coordinate.
    pub y: f32,
    /// The z coordinate.
    pub z: f32,
}

impl Point3 {
    /// Creates a new point.
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    /// Returns whether every coordinate is finite.
    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }

    /// Returns the distance to another point.
    pub fn distance(self, rhs: Self) -> f32 {
        (self - rhs).length()
    }

    /// Returns the midpoint between two points.
    pub fn midpoint(self, rhs: Self) -> Self {
        self + ((rhs - self) * 0.5)
    }
}

impl Add<Vector3> for Point3 {
    type Output = Self;

    fn add(self, rhs: Vector3) -> Self::Output {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

impl Sub<Point3> for Point3 {
    type Output = Vector3;

    fn sub(self, rhs: Point3) -> Self::Output {
        Vector3::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

/// A bone between two distinct finite points.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LineSegment3 {
    /// The start point.
    pub start: Point3,
    /// The end point.
    pub end: Point3,
}

impl LineSegment3 {
    /// Creates a segment, rejecting non-finite or identical endpoints.
    pub fn new(start: Point3, end: Point3) -> Result<Self> {
        if !start.is_finite() || !end.is_finite() {
            return Err(invalid_argument("points must be finite"));
        }
        if start == end {
            return Err(invalid_argument(
                "line segment start and end must not be identical",
            ));
        }
        Ok(Self { start, end })
    }

    /// Returns the segment length.
    pub fn length(self) -> f32 {
        self.start.distance(self.end)
    }

    /// Returns the segment midpoint.
    pub fn midpoint(self) -> Point3 {
        self.start.midpoint(self.end)
    }
}
