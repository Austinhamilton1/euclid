use core::f64;
use std::collections::HashMap;

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
    /*
     * Create a new SpatialHashMap.
     * Arguments:
     *     aabb: AABB - Limits of the SpatialHashMap.
     *     cell_size: Point - How large the cells should be.
     */
    pub fn new(aabb: AABB, cell_size: Point) -> Self {
        let buckets: HashMap<(usize, usize), Vec<&'a Geometry2D<'a>>> = HashMap::new();

        Self { aabb, buckets, cell_size }
    }

    /*
     * Get the minimal AABB for a set of objects.
     * Arguments:
     *     objects: &[Geometry2D<'a>] - Calculate for these objects.
     * Returns:
     *     AABB - The minimal AABB that spans the objects.
     */
    pub fn minimal_aabb(objects: &[Geometry2D<'a>]) -> AABB {
        let mut aabb = AABB::new(Point::new(f64::INFINITY, f64::INFINITY), Point::new(-f64::INFINITY, -f64::INFINITY));
        for object in objects {
            aabb = aabb.union(&object.aabb());
        }
        aabb
    }

    /*
     * Returns the optimal cell size for a predefined set of objects.
     * Arguments:
     *     objects: &[Geometry2D<'a>] - Check these objects.
     * Returns:
     *     Point - The optimal cell size for these objects.
     */
    pub fn optimal_cell_size(objects: &[Geometry2D<'a>]) -> Point {
        let sum_diameters: f64 = objects
            .iter()
            .map(|object| object.aabb().diameter())
            .sum();

        let avg_diameter = sum_diameters / (objects.len() as f64);

        Point::new(avg_diameter * 2.0, avg_diameter * 2.0)
    }

    /*
     * Return the hash for a point in this SpatialHashMap.
     * Arguments:
     *     p: Point - Point to hash.
     * Returns:
     *     (usize, usize) - Index into buckets for the point.
     */
    fn hash(&self, p: Point) -> (usize, usize) {
        let delta = p - self.aabb.min;
        let max_delta = self.aabb.max - self.aabb.min;
        let x: usize = (delta.x / self.cell_size.x).clamp(0.0, max_delta.x / self.cell_size.x).floor() as usize;
        let y: usize = (delta.y / self.cell_size.y).clamp(0.0, max_delta.y / self.cell_size.y).floor() as usize;

        (x, y)
    }

    /*
     * Insert an object into all buckets it crosses.
     * Arguments:
     *     object: &'a Geometry2D<'a> - A reference to the object.
     */
    pub fn insert(&mut self, object: &'a Geometry2D<'a>) {
        let min_hash = self.hash(self.aabb.min);
        let max_hash = self.hash(self.aabb.max);

        for i in min_hash.0..max_hash.0 {
            for j in min_hash.1..max_hash.1 {
                if let Some(bucket) = self.buckets.get_mut(&(i , j)) {
                    bucket.push(object);
                } else {
                    self.buckets.insert((i, j), vec![object]);
                }
            }
        }
    }

    /*
     * Return a vector with all potentially intersecting objects of the query object.
     * Arguments:
     *     query: &'a Geometry2D<'a> - Query this object.
     * Returns:
     *     Vec<&'a Geometry2D<'a>>- A vector of objects if the query is valid, None otherwise.
     */
    pub fn query(&self, query: &'a Geometry2D<'a>) -> Vec<&'a Geometry2D<'a>> {
        let aabb = query.aabb();

        let min_hash = self.hash(aabb.min);
        let max_hash = self.hash(aabb.max);

        let mut objects: Vec<&'a Geometry2D<'a>> = Vec::new();

        for i in min_hash.0..max_hash.0 {
            for j in min_hash.1..max_hash.1 {
                if let Some(bucket) = self.buckets.get(&(i, j)) {
                    for object in bucket {
                        objects.push(object);
                    }
                }
            }
        }

        objects
    }
}