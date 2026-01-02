use core::f64;
use std::{collections::HashMap, f32::INFINITY, hash::Hash};

use super::*;

pub enum Geometry2D<'a> {
    Point(&'a Point),
    Segment(&'a Segment),
    SimplePolygon(&'a SimplePolygon),
    ConvexPolygon(&'a ConvexPolygon),
}

impl<'a> HasAabb for Geometry2D<'a> {
    fn aabb(&self) -> AABB {
        use Geometry2D::*;

        match self {
            Point(p) => p.aabb(),
            Segment(s) => s.aabb(),
            SimplePolygon(p) => p.aabb(),
            ConvexPolygon(p) => p.aabb(),    
        }
    }
}

pub struct SHM<'a> {
    aabb: AABB,
    buckets: HashMap<(usize, usize), Vec<Geometry2D<'a>>>,
    cell_size: Point,
}

impl<'a> SHM<'a> {
    pub fn new(cell_size: Point) -> Self {
        let aabb = AABB::new(Point::new(f64::INFINITY, f64::INFINITY), Point::new(-f64::INFINITY, -f64::INFINITY));
        let buckets: HashMap<(usize, usize), Vec<Geometry2D<'a>>> = HashMap::new();

        Self { aabb, buckets, cell_size }
    }

    pub fn hash(&self, p: Point) -> Option<(usize, usize)> {
        if p.x < self.aabb.min.x || p.x > self.aabb.max.x ||
            p.y < self.aabb.min.y || p.y > self.aabb.max.y {
                None
            } else {
                let delta = p - self.aabb.min;
                let x: usize = (delta.x / self.cell_size.x).floor() as usize;
                let y: usize = (delta.y / self.cell_size.y).floor() as usize;

                Some((x, y))
            }
    }

    pub fn insert(&mut self, object: &Geometry2D<'a>) {
        let aabb = object.aabb();
        self.aabb = self.aabb.union(&aabb);

        let min_hash = self.hash(aabb.min);
        let max_hash = self.hash(aabb.max);
    }
}