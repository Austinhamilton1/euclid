const EPS: f64 = 1e-9;

/*
 * Point with an x and y position.
 */
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

impl Point {
    /*
     * Create a new Point.
     * Arguments:
     *     x: f64
     *     y: f64
     */
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    /*
     * Dot product for Points.
     * Arguments:
     *     other: Point
     * Returns:
     *     f64 - Dot product of two points.
     */
    pub fn dot(self, other: Point) -> f64 {
        self.x*other.x + self.y*other.y
    }

    /*
     * Cross product for Points.
     * Arguments:
     *     other: Point
     * Returns:
     *     f64 - Cross product of two points.
     */
    pub fn cross(self, other: Point) -> f64 {
        self.x*other.y - self.y*other.x
    }
}

impl std::ops::Index<usize> for Point {
    type Output = f64;

    fn index(&self, index: usize) -> &Self::Output {
        if index != 0 && index != 1 {
            panic!("Index {index} out of range for Point");
        }

        if index == 0 { &self.x } else { &self.y }
    }
}

impl std::ops::Add<Point> for Point {
    type Output = Point;

    /*
     * Add two points together.
     */
    fn add(self, rhs: Point) -> Self::Output {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
        }
    }
}

impl std::ops::Sub<Point> for Point {
    type Output = Point;

    /*
     * Subtract a point from another point.
     */
    fn sub(self, rhs: Point) -> Self::Output {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
        }
    }
}

impl std::ops::Mul<Point> for f64 {
    type Output = Point;

    /* 
     * Scalar multiplication for a Point.
     */
    fn mul(self, rhs: Point) -> Self::Output {
        Point {
            x: self * rhs.x,
            y: self * rhs.y,
        }
    }
}

/*
 * Line segment with a start and end Point.
 */
#[derive(Debug, Clone, Copy)]
pub struct Segment {
    pub a: Point,
    pub b: Point,
}

impl Segment {
    /*
     * Create a new Segment.
     * Arguments:
     *     a: Point
     *     b: Point
     */
    pub fn new(a: Point, b: Point) -> Self {
        Self { a, b }
    }
}

/*
 * Ray with a start Point and a direction.
 */
#[derive(Debug, Clone, Copy)]
pub struct Ray {
    p: Point,
    d: Point,
}

impl Ray {
    /*
     * Create a new Ray.
     * Arguments:
     *     p: Point
     *     d: Point 
     */
    pub fn new(p: Point, d: Point) -> Self {
        Self { p, d }
    }
}

/*
 * Axis-Aligned Bounding Box for Points.
 */
#[derive(Debug, Clone, Copy)]
    pub struct AABB {
    pub min: Point,
    pub max: Point,
    pub centroid: Point,
}

impl AABB {
    /*
     * Create a new Axis-Aligned Bounding box from two points.
     * Arguments:
     *     min: Point
     *     max: Point
     */
    pub fn new(min: Point, max: Point) -> Self {
        let centroid = Point {
            x: (min.x + max.x) / 2.0,
            y: (min.y + max.y) / 2.0,
        };

        Self { min, max, centroid }
    }

    /*
     * Create a AABB around a line segment.
     * Arguments:
     *     a: Point
     *     b: Point
     */
    pub fn from_line(a: &Point, b: &Point) -> Self {
        let min_x = if a.x < b.x { a.x } else { b.x };
        let min_y = if a.y < b.y { a.y } else { b.y };
        let max_x = if a.x > b.x { a.x } else { b.x };
        let max_y = if a.y > b.y { a.y } else { b.y };

        let centroid = Point {
            x: (min_x + max_x) / 2.0,
            y: (min_y + max_y) / 2.0,
        };

        Self {
            min: Point::new(min_x, min_y),
            max: Point::new(max_x, max_y),
            centroid,
        }
    }

    /*
     * Create a new Axis-Aligned Bounding box from a collection of points.
     * Arguments:
     *     points: &[Point]
     */
    pub fn from_points(points: &[Point]) -> Self {
        // Ensure a non-empty points list
        assert!(!points.is_empty());

        // Calculate the minimum and maximum values of x and y
        let mut min_x = points[0].x;
        let mut min_y = points[0].y;
        let mut max_x = points[0].x;
        let mut max_y = points[0].y;
        
        for point in points.iter() {
            if point.x < min_x {
                min_x = point.x;
            }

            if point.y < min_y {
                min_y = point.y;
            }

            if point.x > max_x {
                max_x = point.x;
            }

            if point.y > max_y {
                max_y = point.y;
            }
        }

        let centroid = Point {
            x: (min_x + max_x) / 2.0,
            y: (min_y + max_y) / 2.0,
        };

        Self {
            min: Point::new(min_x, min_y),
            max: Point::new(max_x, max_y),
            centroid,
        }
    }

    /*
     * Determine if a point is within the AABB.
     * Arguments:
     *     point: Point - Test this point.
     * Returns:
     *     bool - True if the point is in the AABB, false otherwise.
     */
    pub fn contains(&self, point: Point) -> bool {
        point.x <= self.max.x 
            && point.x >= self.min.x 
            && point.y <= self.max.y 
            && point.y >= self.min.y
    }

    /*
     * Determine if this AABB intersects with another type
     * that also has an AABB.
     * Arguments:
     *     obj: &impl HasAabb - An object that has an AABB.
     * Returns:
     *     bool - True if the object intersects with this AABB, false otherwise.
     */
    pub fn intersects(&self, obj: &impl HasAabb) -> bool {
        // Get the AABB of the other object
        let aabb = obj.aabb();

        // Bounds checks
        !(aabb.max.x < self.min.x 
            || aabb.min.x > self.max.x 
            || aabb.max.y < self.min.y 
            || aabb.min.y > self.max.y)
    }

    /*
     * Merge two AABB's.
     * Arguments:
     *     other: &AABB - The other AABB to merge with this one.
     * Returns:
     *     AABB - A new AABB that is the union of these two AABB's.
     */
    pub fn union(&self, other: &AABB) -> Self {
        let min_x = if self.min.x < other.min.x { self.min.x } else { other.min.x };
        let min_y = if self.min.y < other.min.y { self.min.y } else { other.min.y };
        let max_x = if self.max.x > other.max.x { self.max.x } else { other.max.x };
        let max_y = if self.max.y > other.max.y { self.max.y } else { other.max.y };

        let centroid = Point {
            x: (min_x + max_x) / 2.0,
            y: (min_y + max_y) / 2.0,
        };

        Self {
            min: Point::new(min_x, min_y),
            max: Point::new(max_x, max_y),
            centroid,
        }
    }

    /*
     * Return the diameter of the AABB (diagonal diameter).
     * Returns:
     *     f64 - The diameter of the bounding box.
     */
    pub fn diameter(&self) -> f64 {
        let diagonal = self.max - self.min;
        diagonal.dot(diagonal).sqrt()
    }
}

pub trait HasAabb {
    fn aabb(&self) -> AABB;
}

impl HasAabb for Point {
    /*
     * Get the AABB of a point (trivial - done for generalization).
     */
    fn aabb(&self) -> AABB {
        AABB::new(*self, *self)
    }
}

impl HasAabb for Segment {
    /*
     * Get the AABB of a line segment.
     */
    fn aabb(&self) -> AABB {
        AABB::from_line(&self.a, &self.b)
    }
}

impl HasAabb for AABB {
    /*
     * Get the AABB of an AABB (trivial - done for generalization).
     */
    fn aabb(&self) -> AABB {
        *self
    }
}

pub trait Contains {
    fn contains(&self, other: &Point) -> bool;
}

pub trait Intersects<Rhs = Self> {
    fn intersects(&self, other: &Rhs) -> bool;
}

impl Intersects<Segment> for Segment {
    fn intersects(&self, other: &Segment) -> bool {
        // Bounding box check optimization
        if self.aabb().intersects(&other.aabb()) {
            let p1 = self.a;
            let q1 = self.b;
            let p2 = other.a;
            let q2 = other.b;

            // Get orientation of endpoints
            let o1 = orientation(p1, q1, p2);
            let o2 = orientation(p1, q1, q2);
            let o3 = orientation(p2, q2, p1);
            let o4 = orientation(p2, q2, q1);

            // General intersection case
            if o1 != o2 && o3 != o4 {
                return true;
            }

            // Collinear intersection cases
            if o1 == 0 && on_segment(p1, p2, q1) { return true; }
            if o2 == 0 && on_segment(p1, q2, q1) { return true; }
            if o3 == 0 && on_segment(p2, p1, q2) { return true; }
            if o4 == 0 && on_segment(p2, q1, q2) { return true; }
        }

        false
    }
}

impl Intersects<Segment> for Ray {
    /*
     * Check if a Ray intersects a Segment.
     * Arguments:
     *     other: &Segment
     * Returns:
     *     bool - True if the two intersect, false otherwise.
     */
    fn intersects(&self, other: &Segment) -> bool {
        // Parameterize the equations
        let v = other.b - other.a;
        let w = self.p - other.a;

        // Calculate the cross product
        let denom = v.cross(self.d);

        // Parallel case
        if denom.abs() < EPS {
            // Check collinearity
            if w.cross(v).abs() > EPS {
                return false;
            }

            // Project endpoints onto ray direction
            let t0 = (other.a - self.p).dot(self.d);
            let t1 = (other.b - self.p).dot(self.d);

            return t0 >= 0.0 || t1 >= 0.0;
        }

        // Solve for parameters
        let u = w.cross(self.d) / denom;
        let t = w.cross(v) / denom;

        u >= 0.0 && u <= 1.0 && t >= 0.0
    }
}

impl Intersects<Ray> for Segment {
    /*
     * Check if a Segment intersects a Ray.
     * Arguments:
     *     other: &Ray
     * Returns:
     *     bool - True if the two intersect, false otherwise.
     */
    fn intersects(&self, other: &Ray) -> bool {
        other.intersects(self)
    }
}

pub trait Distance<Rhs = Self> {
    fn distance(&self, other: &Rhs) -> f64;
}

impl Distance<Point> for Point {
    /*
     * Distance between two points.
     * Arguments:
     *     other: &Point
     */
    fn distance(&self, other: &Point) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx*dx + dy*dy).sqrt()
    }
}

impl Distance<Segment> for Point {
    /*
     * Distance between a point and a line segment.
     * Arguments:
     *     other: &Segment
     */
    fn distance(&self, other: &Segment) -> f64 {            
        // Get the line segment's direction
        let v = other.b - other.a;

        // From start of segment to point P
        let w = *self - other.a;

        // Check for divide by zero
        let vv = v.dot(v);
        if vv == 0.0 {
            return self.distance(&other.a);
        }

        // Determine how far along the line (A -> B) the projection of P falls
        let mut t = (w.dot(v)) / vv;

        // Clamp t to the segment
        if t < 0.0 {
            t = 0.0;
        } else if t > 1.0 {
            t = 1.0;
        }

        // Find closest point
        let c = other.a + t * v;

        self.distance(&c)
    }
}

impl Distance<Point> for Segment {
    /*
     * Distance between a line segment and a point.
     * Arguments:
     *     other: &Point
     */
    fn distance(&self, other: &Point) -> f64 {
        other.distance(self)
    }
}

impl Distance<Segment> for Segment {
    /*
     * Distance between two line segments.
     * Arguments:
     *     other: &Segment
     */
    fn distance(&self, other: &Segment) -> f64 {
        // Check if the line segments are intersecting
        if self.intersects(other) { return 0.0; }

        // Return the minimum distance between each endpoint and the other segment
        self.a.distance(other)
            .min(self.b.distance(other))
            .min(other.a.distance(self))
            .min(other.b.distance(self))
    }
}

/* 
 * Get the orientation of three points.
 * Arguments:
 *     p: Point
 *     q: Point
 *     r: Point
 * Returns:
 *     0 - Collinear
 *     1 - Clockwise
 *     2 - Counterclockwise
 */
pub fn orientation(p: Point, q: Point, r: Point) -> i32 {
    let val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y);
    if val.abs() < EPS { 0 }
    else if val > 0.0 { 1 }
    else { 2 }
}

/*
 * Determine if a point is on a line segment.
 * Arguments:
 *     p: Point
 *     q: Point
 *     r: Point
 * Returns:
 *     bool - True if q is on p -> r, false otherwise.
 */
fn on_segment(p: Point, q: Point, r: Point) -> bool {
    q.x >= p.x.min(r.x) && q.x <= p.x.max(r.x) &&
    q.y >= p.y.min(r.y) && q.y <= p.y.max(r.y)
}

/* Geometry will be the basis for all indexed queries */
pub trait Geometry: HasAabb {}

impl Geometry for Point {}
impl Geometry for Segment {}
impl Geometry for AABB {}


/* Unit Tests */


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ops_test() {
        let a = Point::new(5.0, 6.0);
        let b = Point::new(1.0, -1.0);

        let add = a + b;
        let sub = a - b;
        let mul = 3.0 * b;

        assert!((6.0 - add.x).abs() < EPS && (5.0 - add.y).abs() < EPS);
        assert!((4.0 - sub.x).abs() < EPS && (7.0 - sub.y).abs() < EPS);
        assert!((3.0 - mul.x).abs() < EPS && (-3.0 - mul.y).abs() < EPS);
    }

    #[test]
    fn dot_test() {
        let a = Point::new(5.0, 6.0);
        let b = Point::new(-1.0, 1.0);
        let c = a.dot(b);
        assert!((1.0 - c).abs() < EPS);
    }

    #[test]
    fn cross_test() {
        let a = Point::new(1.0, 1.0);
        let b = Point::new(-1.0, -1.0);
        let c = a.cross(b);
        assert!(c.abs() < EPS);
    }

    #[test]
    fn ray_segment_intersect_test() {
        let segment = Segment::new(Point::new(10.0, 0.0), Point::new(0.0, 10.0));
        let ray = Ray::new(Point::new(0.0, 0.0), Point::new(1.0, 1.0));

        assert!(ray.intersects(&segment));
        assert!(segment.intersects(&ray));

        let segment = Segment::new(Point::new(0.0, 10.0), Point::new(10.0, 0.0));
        let ray = Ray::new(Point::new(0.0, 0.0), Point::new(-1.0, 1.0));

        assert!(!ray.intersects(&segment));
        assert!(!segment.intersects(&ray));
    }

    #[test]
    fn aabb_contains_test() {
        let point = Point::new(0.0, 0.0);
        let shape = vec![Point::new(-1.0, -1.0), Point::new(1.0, -1.0), Point::new(0.0, 1.0)];
        let aabb = AABB::from_points(&shape);

        assert!(aabb.contains(point));

        let point = Point::new(5.0, 6.3);
        
        assert!(!aabb.contains(point));
    }

    #[test]
    fn aabb_intersect_test() {
        let triangle = vec![Point::new(-3.0, 2.0), Point::new(1.0, 1.0), Point::new(-1.0, 3.0)];
        let rect = vec![Point::new(-2.0, 1.0), Point::new(-1.0, 1.0), Point::new(-1.0, 2.0), Point::new(-2.0, 2.0)];
        let tri_aabb = AABB::from_points(&triangle);
        let rect_aabb = AABB::from_points(&rect);

        assert!(rect_aabb.intersects(&tri_aabb));

        let triangle = vec![Point::new(0.0, 2.0), Point::new(4.0, 1.0), Point::new(2.0, 3.0)];
        let tri_aabb = AABB::from_points(&triangle);

        assert!(!rect_aabb.intersects(&tri_aabb));
    }

    #[test]
    fn aabb_union_test() {
        let triangle = vec![Point::new(-3.0, 2.0), Point::new(1.0, 1.0), Point::new(-1.0, 3.0)];
        let rect = vec![Point::new(-2.0, 1.0), Point::new(-1.0, 1.0), Point::new(-1.0, 2.0), Point::new(-2.0, 2.0)];
        let tri_aabb = AABB::from_points(&triangle);
        let rect_aabb = AABB::from_points(&rect);

        let aabb = rect_aabb.union(&tri_aabb);

        assert!(
            (aabb.min.x - -3.0).abs() < EPS && 
            (aabb.min.y - 1.0).abs() < EPS && 
            (aabb.max.x - 1.0).abs() < EPS &&
            (aabb.max.y - 3.0).abs() < EPS
        );
    }

    #[test]
    fn segment_segment_intersect_test() {
        let s1 = Segment::new(Point::new(10.0, 0.0), Point::new(0.0, 10.0));
        let s2 = Segment::new(Point::new(0.0, 0.0), Point::new(10.0, 10.0));

        assert!(s1.intersects(&s2));

        let s2 = Segment::new(Point::new(15.0, -5.0), Point::new(5.0, 5.0));
        assert!(s1.intersects(&s2));

        let s2 = Segment::new(Point::new(-4.0, -4.0), Point::new(-6.0, -6.0));
        assert!(!s1.intersects(&s2));

        let s2 = Segment::new(Point::new(-5.0, 15.0), Point::new(-10.0, 20.0));
        assert!(!s1.intersects(&s2));
    }

    #[test]
    fn point_point_distance_test() {
        let p1 = Point::new(0.0, 0.0);
        let p2 = Point::new(3.0, 4.0);

        let dist = p1.distance(&p2);

        assert!((dist - 5.0).abs() < EPS);
    }

    #[test]
    fn point_segment_distance_test() {
        let segment = Segment::new(Point::new(10.0, 0.0), Point::new(0.0, 10.0));
        let p1 = Point::new(0.0, 0.0);
        let p2 = Point::new(-3.0, 14.0);
        let p3 = Point::new(22.0, 5.0);

        let d1 = p1.distance(&segment);
        let d2 = p2.distance(&segment);
        let d3 = p3.distance(&segment);

        assert!((d1 - 7.071067812).abs() < EPS);
        assert!((d2 - 5.0).abs() < EPS);
        assert!((d3 - 13.0).abs() < EPS);
    }

    #[test]
    fn segment_segment_distance_test() {
        let s1 = Segment::new(Point::new(10.0, 0.0), Point::new(0.0, 10.0));
        let s2 = Segment::new(Point::new(-5.0, -5.0), Point::new(0.0, 0.0));
        let s3 = Segment::new(Point::new(10.0, -3.0), Point::new(20.0, -10.0));

        let d1 = s2.distance(&s1);
        let d2 = s3.distance(&s1);

        assert!((d1 - 7.071067812).abs() < EPS);
        assert!((d2 - 3.0).abs() < EPS);
    }

    #[test]
    fn orientation_test() {
        let a = Point::new(-1.0, 0.0);
        let b = Point::new(1.0, 0.0);
        let c = Point::new(0.0, 1.0);
        let d = Point::new(2.0, 0.0);

        assert!(orientation(a, b, c) == 2);
        assert!(orientation(a, c, b) == 1);
        assert!(orientation(a, b, d) == 0);
    }
}