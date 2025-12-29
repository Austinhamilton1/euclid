use super::{Point, Segment, AABB, HasAabb, Contains, Intersects, Distance, Geometry, orientation};

pub struct SimplePolygon {
    vertices: Vec<Point>,
}

impl SimplePolygon {
    /*
     * Create a new empty polygon.
     */
    pub fn new() -> Self {
        Self {
            vertices: Vec::new(),
        }
    }

    /*
     * Create a polygon from a set of points.
     * Arguments:
     *     points: &[Point]
     */
    pub fn from_points(points: &[Point]) -> Self {
        Self {
            vertices: points.to_vec(),
        }
    }

    /*
     * Add a new point to the polygon.
     * Arguments:
     *     point: Point
     */
    pub fn add_point(&mut self, point: Point) {
        self.vertices.push(point);
    }

    pub fn convex_hull(&self) -> ConvexPolygon {
        if self.vertices.len() <= 2 {
            return ConvexPolygon { vertices: self.vertices.clone() };
        }

        // Find the pivot: lowest y, then lowest x
        let pivot = self.vertices
            .iter()
            .min_by(|a, b| {
                a.y.partial_cmp(&b.y)
                    .unwrap()
                    .then_with(|| a.x.partial_cmp(&b.x).unwrap())  
            })
            .unwrap();

        // Sort points by polar angle around pivot
        let mut sorted: Vec<Point> = self.vertices
            .iter()
            .copied()
            .filter(|p| *p != *pivot)
            .collect();

        sorted.sort_by(|a, b| {
            let o = orientation(*pivot, *a, *b);

            if o == 0 {
                a.dot(*a)
                    .partial_cmp(&b.dot(*b))
                    .unwrap()
            } else if o > 0 {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Greater
            }
        });

        // Build hull using a stack
        let mut hull: Vec<Point> = Vec::new();
        hull.push(*pivot);
        hull.push(sorted[0]);

        for &p in sorted.iter().skip(1) {
            while hull.len() >= 2 {
                let q = hull[hull.len() - 1];
                let r = hull[hull.len() - 2];

                if orientation(r, q, p) > 0 {
                    break;
                }
                hull.pop();
            }
            hull.push(p);
        }

        ConvexPolygon::from_points(&hull)
    }
}

pub struct ConvexPolygon {
    vertices: Vec<Point>,
}

impl ConvexPolygon {
    /*
     * Create a new empty ConvexPolygon.
     */
    pub fn new() -> Self {
        Self {
            vertices: Vec::new(),
        }
    }

    /*
     * Create and populate a ConvexPolygon.
     * Arguments:
     *     points: &[Point]
     */
    pub fn from_points(points: &[Point]) -> Self {
        Self {
            vertices: points.to_vec(),
        }
    }

    /*
     * Add a point to a ConvexPolygon.
     * Arguments:
     *     point: Point
     */
    pub fn add_point(&mut self, point: Point) {
        self.vertices.push(point);
    }
}

impl HasAabb for SimplePolygon {
    /*
     * Generate the AABB for a SimplePolygon.
     */
    fn aabb(&self) -> AABB {
        AABB::from_points(&self.vertices)
    }
}

impl HasAabb for ConvexPolygon {
    /*
     * Generate the AABB for a ConvexPolygon.
     */
    fn aabb(&self) -> AABB {
        AABB::from_points(&self.vertices)
    }
}

impl Contains for SimplePolygon {
    fn contains(&self, other: &Point) -> bool {
        assert!(self.vertices.len() >= 3);

        // Bounding Box optimization
        if self.aabb().contains(*other) {

        }

        false
    }
}

impl Contains for ConvexPolygon {
    fn contains(&self, other: &Point) -> bool {
        assert!(self.vertices.len() >= 3);

        // Bounding Box optimization
        if self.aabb().contains(*other) {
            // Calculate initial orientation
            let p = self.vertices[0];
            let mut r = self.vertices[1];
            let o = orientation(p, *other, r);

            // Ensure the orientation is the same for every edge
            for (i, &point) in self.vertices.iter().enumerate().skip(1) {
                r = self.vertices[(i + 1) % self.vertices.len()];
                if orientation(point, *other, r) != o {
                    return false;
                }
            }

            return true;
        }
        
        false
    }
}

impl Intersects<Segment> for SimplePolygon {
    /*
     * Check for the intersection of a Segment with a SimplePolygon.
     * Arguments:
     *     other: &Segment
     * Returns:
     *     bool - True if the two intersect, false otherwise.
     */
    fn intersects(&self, other: &Segment) -> bool {
        if self.aabb().intersects(other) {

        }

        false        
    }
}

impl Intersects<SimplePolygon> for Segment {
    /*
     * Check for the intersection of a Segment with a SimplePolygon.
     * Arguments:
     *     other: &SimplePolygon
     * Returns:
     *     bool - True if the two intersect, false otherwise.
     */
    fn intersects(&self, other: &SimplePolygon) -> bool {
        other.intersects(self)
    }
}

impl Intersects<Segment> for ConvexPolygon {
    /*
     * Check for the intersection of a Segment with a ConvexPolygon.
     * Arguments:
     *     other: &Segment
     * Returns:
     *     bool - True if the two intersect, false otherwise.
     */
    fn intersects(&self, other: &Segment) -> bool {
        if self.aabb().intersects(other) {

        }

        false
    }
}

impl Intersects<ConvexPolygon> for Segment {
    /*
     * Check for the intersection of a Segment with a ConvexPolygon.
     * Arguments:
     *     other: &ConvexPolygon
     * Returns:
     *     bool - True if the two intersect, false otherwise.
     */
    fn intersects(&self, other: &ConvexPolygon) -> bool {
        other.intersects(self)
    }
}

impl Intersects<SimplePolygon> for SimplePolygon {
    /*
     * Check for the intersection of a SimplePolygon with a SimplePolygon.
     * Arguments:
     *     other: &SimplePolygon
     * Returns:
     *     bool - True if the two intersect, false otherwise.
     */
    fn intersects(&self, other: &SimplePolygon) -> bool {
        false
    }
}

impl Intersects<ConvexPolygon> for SimplePolygon {
    /*
     * Check for the intersection of a ConvexPolygon with a SimplePolygon.
     * Arguments:
     *     other: &ConvexPolygon
     * Returns:
     *     bool - True if the two intersect, false otherwise.
     */
    fn intersects(&self, other: &ConvexPolygon) -> bool {
        false
    }
}

impl Intersects<SimplePolygon> for ConvexPolygon {
    /*
     * Check for the intersection of a ConvexPolygon with a SimplePolygon.
     * Arguments:
     *     other: &SimplePolygon
     * Returns:
     *     bool - True if the two intersect, false otherwise.
     */
    fn intersects(&self, other: &SimplePolygon) -> bool {
        other.intersects(self)
    }
}

impl Intersects<ConvexPolygon> for ConvexPolygon {
    /*
     * Check for the intersection of a ConvexPolygon with a ConvexPolygon.
     * Arguments:
     *     other: &ConvexPolygon
     * Returns:
     *     bool - True if the two intersect, false otherwise.
     */
    fn intersects(&self, other: &ConvexPolygon) -> bool {
        false
    }
}

impl Geometry for SimplePolygon {}
impl Geometry for ConvexPolygon {}