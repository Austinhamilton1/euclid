use core::f64;
use std::{collections::HashMap};

use super::*;

pub enum Geometry2D<'a> {
    Point(&'a Point),
    Segment(&'a Segment),
    SimplePolygon(&'a SimplePolygon),
    ConvexPolygon(&'a ConvexPolygon),
}

impl<'a> HasAabb for Geometry2D<'a> {
    /*
     * Get the AABB for a Geometry2D.
     */
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

pub struct SpatialHashMap<'a> {
    aabb: AABB,
    buckets: HashMap<(usize, usize), Vec<&'a Geometry2D<'a>>>,
    cell_size: Point,
}

impl<'a> SpatialHashMap<'a> {
    pub fn new(cell_size: Point) -> Self {
        let aabb = AABB::new(Point::new(f64::INFINITY, f64::INFINITY), Point::new(-f64::INFINITY, -f64::INFINITY));
        let buckets: HashMap<(usize, usize), Vec<&'a Geometry2D<'a>>> = HashMap::new();

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

    /*
     * Insert an object into all buckets it crosses.
     * Arguments:
     *     object: &'a Geometry2D<'a> - A reference to the object.
     */
    pub fn insert(&mut self, object: &'a Geometry2D<'a>) {
        let aabb = object.aabb();
        self.aabb = self.aabb.union(&aabb);

        let min_hash = self.hash(aabb.min);
        let max_hash = self.hash(aabb.max);

        if let Some(min) = min_hash {
            if let Some(max) = max_hash {
                for i in min.0..max.0 {
                    for j in min.1..max.1 {
                        if let Some(bucket) = self.buckets.get_mut(&(i , j)) {
                            bucket.push(object);
                        } else {
                            self.buckets.insert((i, j), vec![object]);
                        }
                    }
                }
            }
        }
    }

    /*
     * Return a vector with all potentially intersecting objects of the query object.
     * Arguments:
     *     query: &'a Geometry2D<'a> - Query this object.
     * Returns:
     *     Option<Vec<&'a Geometry2D<'a>>> - A vector of objects if the query is valid, None otherwise.
     */
    pub fn query(&self, query: &'a Geometry2D<'a>) -> Option<Vec<&'a Geometry2D<'a>>> {
        let aabb = query.aabb();

        let min_hash = self.hash(aabb.min);
        let max_hash = self.hash(aabb.max);

        let mut objects: Vec<&'a Geometry2D<'a>> = Vec::new();

        if let Some(min) = min_hash {
            if let Some(max) = max_hash {
                for i in min.0..max.0 {
                    for j in min.1..max.1 {
                        if let Some(bucket) = self.buckets.get(&(i, j)) {
                            for object in bucket {
                                objects.push(object);
                            }
                        }
                    }
                }

                return Some(objects);
            }
        }

        None
    }
}