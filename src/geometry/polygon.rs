use core::f64;

use super::*;

const EPS: f64 = 1e-9;

pub struct SimplePolygon {
    center: Point,
    vertices: Vec<Point>,
}

impl SimplePolygon {
    /*
     * Create a new empty polygon.
     */
    pub fn new() -> Self {
        Self {
            center: Point::new(0.0, 0.0),
            vertices: Vec::new(),
        }
    }

    /*
     * Create a polygon from a set of points.
     * Arguments:
     *     points: &[Point]
     */
    pub fn from_points(points: &[Point]) -> Self {
        // Calculate the center of the points
        let mut center = Point::new(0.0, 0.0);
        for &point in points.iter() {
            center = center + point;
        }

        let len = points.len() as f64;

        center = (1.0 / len) * center;

        Self {
            center,
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

        // Recalculate the center of the points
        let mut center = Point::new(0.0, 0.0);
        for &point in self.vertices() {
            center = center + point;
        }

        let len = self.vertices.len() as f64;

        center = (1.0 / len) * center;

        self.center = center;
    }

    pub fn convex_hull(&self) -> ConvexPolygon {
        assert!(self.vertices.len() >= 3);

        // Find the pivot: lowest y, then lowest x
        let pivot = self.vertices()
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
                pivot.dot(*a)
                    .partial_cmp(&pivot.dot(*b))
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

                if orientation(r, q, p) == 1 {
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
    center: Point,
    vertices: Vec<Point>,
}

impl ConvexPolygon {
    /*
     * Create a new empty ConvexPolygon.
     */
    pub fn new() -> Self {
        Self {
            center: Point::new(0.0, 0.0),
            vertices: Vec::new(),
        }
    }

    /*
     * Create and populate a ConvexPolygon.
     * Arguments:
     *     points: &[Point]
     */
    pub fn from_points(points: &[Point]) -> Self {
        // Calculate the center of the points
        let mut center = Point::new(0.0, 0.0);
        for &point in points.iter() {
            center = center + point;
        }

        let len = points.len() as f64;

        center = (1.0 / len) * center;

        Self {
            center,
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

        // Recalculate the center of the points
        let mut center = Point::new(0.0, 0.0);
        for &point in self.vertices() {
            center = center + point;
        }

        let len = self.vertices.len() as f64;

        center = (1.0 / len) * center;

        self.center = center;
    }
}

pub struct LineEdgeIter<'a> {
    verts: std::slice::Iter<'a, Point>,
    prev: Option<&'a Point>,
}

impl<'a> LineEdgeIter<'a> {
    /*
     * Create a new LineEdgeIter with the vertices from a line.
     * Arguments:
     *     vertices: &'a [Point]
     */
    pub fn new(vertices: &'a [Point]) -> Self {
        let mut iter = vertices.iter();
        let prev = iter.next();

        Self {
            verts: iter,
            prev,
        }
    }
}

impl<'a> Iterator for LineEdgeIter<'a> {
    type Item = (&'a Point, &'a Point);

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(curr) = self.verts.next() {
            let edge = Some((self.prev.unwrap(), curr));
            self.prev = Some(curr);
            return edge;
        }

        None
    }
}

pub struct PolygonEdgeIter<'a> {
    verts: std::slice::Iter<'a, Point>,
    first: Option<&'a Point>,
    prev: Option<&'a Point>,
    done: bool,
}

impl<'a> PolygonEdgeIter<'a> {
    /*
     * Create a new PolygonEdgeIter with the vertices from a Polygon.
     * Arguments:
     *     vertices: &'a [Point]
     */
    pub fn new(vertices: &'a [Point]) -> Self {
        let mut iter = vertices.iter();

        let first = iter.next();
        let prev = first;

        Self {
            verts: iter,
            first,
            prev,
            done: false,
        }
    }

    /*
     * Generate normals for the edges.
     * Returns:
     *     impl Iterator<Item = Point> + 'a - An iterator of Points (normal vectors).
     */
    pub fn normals(self) -> impl Iterator<Item = Point> + 'a {
        self.map(|(a, b)| {
            let edge = *b - *a;
            Point::new(-edge.y, edge.x)
        })
    }
}

impl<'a> Iterator for PolygonEdgeIter<'a> {
    type Item = (&'a Point, &'a Point);

    fn next(&mut self) -> Option<Self::Item> {
        // Normal edges
        if let Some(curr) = self.verts.next() {
            let edge = Some((self.prev.unwrap(), curr));
            self.prev = Some(curr);
            return edge;
        }

        // Closing edge (last -> first)
        if !self.done {
            self.done = true;

            if let (Some(last), Some(first)) = (self.prev, self.first) {
                if last as *const _ != first as *const _ {
                    return Some((last, first));
                }
            }
        }

        None
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
            // Need to count how many times a ray pointing out of the Polygon crosses the shape
            let mut intersect_count = 0;
            let test_ray = Ray::new(*other, Point::new(1.0, 0.0));

            self.edges().for_each(|(&p, &r)| {
                let s = Segment::new(p, r);
                if test_ray.intersects(&s) {
                    intersect_count += 1;
                }
            });

            // If the ray intersects an odd count, the point is in the Polygon,
            // otherwise its outside
            return intersect_count % 2 == 1;
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
            let r = self.vertices[1];
            let o = orientation(p, *other, r);

            // Ensure the orientation is the same for every edge
            for (&p, &r) in self.edges().skip(1) {
                if orientation(p, *other, r) != o {
                    return false;
                }
            }

            return true;
        }
        
        false
    }
}

/*
 * Brute force algorithm for determining intersection of two generic shapes.
 * Arguments:
 *     a: &impl Shape2D
 *     b: &impl Shape2D
 * Returns:
 *     bool - True if the objects intersect, false otherwise.
 */
fn brute_force_intersects(a: &impl Shape2D, b: &impl Shape2D) -> bool {
    // O(n^2) brute force method. Check every edge against every other edge.
    for (&p1, &p2) in a.edges() {
        for (&p3, &p4) in b.edges() {
            let s1 = Segment::new(p1, p2);
            let s2 = Segment::new(p3, p4);
            if s1.intersects(&s2) {
                return true;
            }
        }
    }

    false
}

/*
 * Brute force algorithm for calculating distance between two generic shapes.
 * Arguments:
 *     a: &impl Shape2D
 *     b: &impl Shape2D
 * Returns:
 *     f64 - The distance between the objects.
 */
fn brute_force_distance(a: &impl Shape2D, b: &impl Shape2D) -> f64 {
    // O(n^2) brute force method. Check every edge against every other edge.
    a.edges()
        .map(|(&p1, &p2)| {
            let s1 = Segment::new(p1, p2);
            b.edges()
                .map(|(&p3, &p4)| {
                    let s2 = Segment::new(p3, p4);
                    s2.distance(&s1)
                })
                .min_by(|a, b| {
                    a.partial_cmp(b)
                        .unwrap()
                })
                .unwrap()
        })
        .min_by(|a, b| {
            a.partial_cmp(b)
                .unwrap()  
        })
            .unwrap()
}

/*
 * GJK algorithm for determining intersection of two convex polygons.
 * Arguments:
 *     a: &impl Shape2D
 *     b: &impl Shape2D
 * Returns:
 *     bool - True if the objects intersect, false otherwise.
 */
fn gjk_intersects(a: &impl Shape2D, b: &impl Shape2D) -> bool {
    // Initial direction should be the vector of separation between
    // the centers of the shapes
    let mut dir = a.center().unwrap() - b.center().unwrap();
    if dir.x == 0.0 && dir.y == 0.0 {
        dir = Point::new(1.0, 0.0);
    }

    // Build the initial simplex
    let mut simplex: Vec<Point> = Vec::with_capacity(3);
    simplex.push(support(a, b, dir));
    dir = Point::new(-simplex[0].x, -simplex[0].y);

    // In a loop, successively build simplexes until one is found that contains
    // the origin (intersects) or one goes beyond the origin (does not intersect).
    loop {
        let p = support(a, b, dir);
        if p.dot(dir) <= 0.0 {
            return false;
        }

        simplex.push(p);

        if handle_simplex_intersection(&mut simplex, &mut dir) {
            return true;
        }
    }
}

/*
 * Handle a simplex from the GJK algorithm below.
 * Arguments:
 *     simplex: &mut Vec<Point> - Handle this simplex.
 *     dir: &mut Point - The direction vector for the support function.
 * Returns:
 *     bool - True if the simplex contains the origin, false otherwise.
 */
fn handle_simplex_intersection(simplex: &mut Vec<Point>, dir: &mut Point) -> bool {
    match simplex.len() {
        2 => handle_line_intersection(simplex, dir),
        3 => handle_triangle_intersection(simplex, dir),
        _ => unreachable!(),
    }
}

/*
 * Helper function for handle_simplex (line case).
 */
fn handle_line_intersection(simplex: &mut Vec<Point>, dir: &mut Point) -> bool {
    let a = simplex[1];
    let b = simplex[0];

    let ab = b - a;

    let ao = Point::new(-a.x, -a.y);

    if ab.dot(ao) > 0.0 {
        *dir = triple_cross(ab, ao, ab);
    } else {
        simplex.clear();
        simplex.push(a);
        *dir = ao;
    }

    false
}

/*
 * Helper function for handle_simplex (triangle case).
 */
fn handle_triangle_intersection(simplex: &mut Vec<Point>, dir: &mut Point) -> bool {
    let a = simplex[2];
    let b = simplex[1];
    let c = simplex[0];

    let ab = b - a;
    let ac = c - a;
    let ao = Point::new(-a.x, -a.y);

    *dir = Point::new(0.0, 0.0);

    // Determine Voronoi region
    let ab_perp = triple_cross(ac, ab, ab);
    if ab_perp.dot(ao) > 0.0 {
        // Origin is outside AB edge
        simplex.remove(0); // remove C
        *dir = ab_perp;
        return false;
    }
    
    let ac_perp = triple_cross(ab, ac, ac);
    if ac_perp.dot(ao) > 0.0 {
        // Origin is outside AC edge
        simplex.remove(1); // remove B
        *dir = ac_perp;
        return false
    }

    true
}

/*
 * GJK algorithm for determining the distance between two convex polygons.
 * Arguments:
 *     a: &impl Shape2D
 *     b: &impl Shape2D
 * Returns:
 *     f64 - The distance between the objects.
 */
fn gjk_distance(a: &impl Shape2D, b: &impl Shape2D) -> f64 {
    0.0
}

/* Helper function for GJK distance. */

/*
 * Perform a triple cross product of three vectors.
 * Arguments:
 *     a: Point,
 *     b: Point,
 *     c: Point,
 * Returns:
 *     Point - The triple cross of a x (b x c).
 */
fn triple_cross(a: Point, b: Point, c: Point) -> Point {
    let ac = a.dot(c);
    let bc = b.dot(c);
    Point {
        x: b.x * ac - a.x * bc,
        y: b.y * ac - a.y * bc,
    }
}

/*
 * Helper function for calculating support functions.
 */
fn support(a: &impl Shape2D, b: &impl Shape2D, dir: Point) -> Point {
    let p1 = a.support(dir).unwrap();
    let p2 = b.support(Point::new(-dir.x, -dir.y)).unwrap();
    p1 - p2
}

impl<T, U> Intersects<T> for U
    where 
        T: Shape2D,
        U: Shape2D,
    {
        fn intersects(&self, other: &T) -> bool {
            // AABB optimization
            if self.aabb().intersects(other) {
                return match self.is_convex() {
                    true => gjk_intersects(self, other),
                    false => brute_force_intersects(self, other),
                }
            }
            
            false
        }
    }

impl<T> Distance<Point> for T
    where 
        T: Shape2D,
    {
        /*
         * Calculate the distance between a Shape2D and a Point.
         * Arguments:
         *     other: &Point
         * Returns:
         *     f64 - The distance between the two objects.
         */
        fn distance(&self, other: &Point) -> f64 {
            // Convex shapes are much easier for points
            if self.is_convex() {
                // Calculate the maximal point in the direction of the other point
                let dir = *other - self.center().unwrap();
                let p1 = self.support(dir).unwrap();
                
                // Need to check if the point is closer to a segment
                let dir = *other - p1;
                let p2 = self.support(dir).unwrap();

                // If these two are equal, closest point is that point
                if p1 == p2 {
                    return p1.distance(other);
                }

                // If these two are not equal, closest point is on the segment connecting 
                // the two
                let s = Segment::new(p1, p2);
                return s.distance(other);
            }

            // Brute force distance
            self.edges()
                .map(|(&p1, &p2)| {
                    let s = Segment::new(p1, p2);
                    s.distance(other)
                })
                .min_by(|a, b| {
                    a.partial_cmp(b)
                        .unwrap()
                })
                .unwrap()
        }
    }

impl<T> Distance<T> for Point 
    where 
        T: Shape2D,
    {
        fn distance(&self, other: &T) -> f64 {
            other.distance(self)
        }
    }

impl<T> Distance<Segment> for T
    where 
        T: Shape2D,
    {
        /*
         * Calculate the distance between a Shape and a Segment.
         * Arguments:
         *     other: &Segment
         * Returns:
         *     f64 - The distance between the two objects.
         */
        fn distance(&self, other: &Segment) -> f64 {
            self.edges()
                .map(|(&p1, &p2)| {
                    let s = Segment::new(p1, p2);
                    s.distance(other)
                })
                .min_by(|a, b| {
                    a.partial_cmp(b)
                        .unwrap()
                })
                .unwrap()
        }
    }

impl<T> Distance<T> for Segment
    where 
        T: Shape2D,
    {
        /*
         * Calculate the distance between a Segment and a Shape.
         * Arguments:
         *     other: &Shape2D
         * Returns:
         *     f64 - The distance between the two objects.
         */
        fn distance(&self, other: &T) -> f64 {
            other.distance(self)
        }
    }

impl<T, U> Distance<T> for U 
    where 
        T: Shape2D,
        U: Shape2D,
    {
        /*
         * Calculate the distance between two shapes.
         * Arguments:
         *     other: Shape2D
         * Returns:
         *     f64 - The distance between the two shapes.
         */
        fn distance(&self, other: &T) -> f64 {
            if self.is_convex() && other.is_convex() {
                return gjk_distance(self, other);
            }

            brute_force_distance(self, other)
        }
    }

impl Geometry for SimplePolygon {}
impl Geometry for ConvexPolygon {}

pub trait Shape2D: Geometry {
    type NodeIter<'a>: Iterator<Item = &'a Point>
        where 
            Self: 'a;

    type EdgeIter<'b>: Iterator<Item = (&'b Point, &'b Point)>
        where 
            Self: 'b;

    fn vertices(&self) -> Self::NodeIter<'_>;
    fn edges(&self) -> Self::EdgeIter<'_>;
    fn is_convex(&self) -> bool;

    /* These methods only apply to convex shapes */
    fn support(&self, _dir: Point) -> Option<Point> {
        None
    }

    fn center(&self) -> Option<Point> {
        None
    }
}

impl Shape2D for SimplePolygon {
    type NodeIter<'a> = std::slice::Iter<'a, Point>
        where
            Self: 'a;
    
    type EdgeIter<'a> = PolygonEdgeIter<'a>
        where 
            Self: 'a;

    /*
     * Return an iterator to this shape's vertices.
     * Returns:
     *     Self::NodeIter<'_> - Non-copying iterator.
     */
    fn vertices(&self) -> Self::NodeIter<'_> {
        self.vertices.iter()
    }

    /*
     * Returns an iterator to this shape's edges.
     * Returns:
     *     PolygonEdgeIter - Non-copying iterator.
     */
    fn edges(&self) -> Self::EdgeIter<'_> {
        PolygonEdgeIter::new(&self.vertices)
    }

    /*
     * Is this shape convex?
     * Returns:
     *     bool
     */
    fn is_convex(&self) -> bool {
        false
    }
}

impl Shape2D for ConvexPolygon {
    type NodeIter<'a> = std::slice::Iter<'a, Point>
        where
            Self: 'a;

    type EdgeIter<'a> = PolygonEdgeIter<'a>
        where 
            Self: 'a;

    /*
     * Return an iterator to this shape's vertices.
     * Returns:
     *     Self::NodeIter<'_> - Non-copying iterator.
     */
    fn vertices(&self) -> Self::NodeIter<'_> {
        self.vertices.iter()
    }

    /*
     * Returns an iterator to this shape's edges.
     * Returns:
     *     PolygonEdgeIter - Non-copying iterator.
     */
    fn edges(&self) -> Self::EdgeIter<'_> {
        PolygonEdgeIter::new(&self.vertices)
    }

    /*
     * Is this shape convex?
     * Returns:
     *     bool
     */
    fn is_convex(&self) -> bool {
        true
    }

    /*
     * Center function for grabbing the center of the Polygon.
     * Returns:
     *     Option<Point> - The center of mass.
     */
    fn center(&self) -> Option<Point> {
        Some(self.center)
    }

    /*
     * Support function for determining the maximal point in a specified direction.
     * Arguments:
     *     dir: Point
     * Returns:
     *     Option<Point> - The maximal point on the ConvexPolygon in the specified direction.
     */
    fn support(&self, dir: Point) -> Option<Point> {
        // Initialize the support point as the first point
        let mut support_point = self.vertices[0];
        let mut maximum_direction = support_point.dot(dir);

        for &point in self.vertices.iter().skip(1) {
            // Since the vertices are ordered and convex, they are monotonically increasing
            // The first decrease means the maximal point has been found
            let new_direction = point.dot(dir);
            if new_direction < maximum_direction {
                break;
            }

            support_point = point;
            maximum_direction = new_direction;
        }

        Some(support_point)
    }
}
/* Unit Tests */

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-9;

    #[test]
    fn convex_hull_test() {
        let points = vec![Point::new(0.0, 0.0), Point::new(2.0, 0.0), Point::new(1.0, 1.0), Point::new(0.0, 2.0)];
        let sp = SimplePolygon::from_points(&points);
        let cp = sp.convex_hull();

        let mut iter = cp.vertices();

        let p1 = iter.next().unwrap();
        let p2 = iter.next().unwrap();
        let p3 = iter.next().unwrap();

        assert!((p1.x - 0.0).abs() < EPS && (p1.y - 0.0).abs() < EPS);
        assert!((p2.x - 0.0).abs() < EPS && (p2.y - 2.0).abs() < EPS);
        assert!((p3.x - 2.0).abs() < EPS && (p3.y - 0.0).abs() < EPS);
    }

    #[test]
    fn simple_polygon_contains_test() {
        let p1 = Point::new(-1.0, 2.5);
        let p2 = Point::new(-3.0, 2.0);
        let vertices = vec![Point::new(-1.0, 1.0), Point::new(-1.0, 2.0), Point::new(1.0, 1.0), Point::new(-1.0, 3.0), Point::new(-3.0, 2.0), Point::new(-2.0, 2.0), Point::new(-2.0, 1.0)];    
        let polygon = SimplePolygon::from_points(&vertices);

        assert!(polygon.contains(&p1));
        assert!(!polygon.contains(&p2));
    }

    #[test]
    fn convex_polygon_contains_test() {
        let p1 = Point::new(-1.0, 2.5);
        let p2 = Point::new(-3.0, 2.0);

        let vertices = vec![Point::new(-1.0, 1.0), Point::new(-1.0, 2.0), Point::new(1.0, 1.0), Point::new(-1.0, 3.0), Point::new(-3.0, 2.0), Point::new(-2.0, 2.0), Point::new(-2.0, 1.0)]; 
        let polygon = SimplePolygon::from_points(&vertices);
        let convex_polygon = polygon.convex_hull();

        assert!(convex_polygon.contains(&p1));
        assert!(!convex_polygon.contains(&p2));
    }

    #[test]
    fn convex_polygon_intersects_test() {
        let shape1 = vec![Point::new(-2.0, 1.0), Point::new(-2.0, 2.0), Point::new(-1.0, 2.0), Point::new(-1.0, 1.0)];
        let shape2 = vec![Point::new(1.0, 1.0), Point::new(-3.0, 2.0), Point::new(-1.0, 3.0)];
        let shape3 = vec![Point::new(1.0, 2.0), Point::new(-3.0, 3.0), Point::new(-1.0, 4.0)];

        let p1 = SimplePolygon::from_points(&shape1);
        let p2 = SimplePolygon::from_points(&shape2);
        let p3 = SimplePolygon::from_points(&shape3);

        let cp1 = p1.convex_hull();
        let cp2 = p2.convex_hull();
        let cp3 = p3.convex_hull();

        assert!(cp1.intersects(&cp2));
        assert!(!cp1.intersects(&cp3));
    }

    #[test]
    fn non_convex_polygon_intersects_test() {
        let shape1 = vec![Point::new(-2.0, 1.0), Point::new(-2.0, 2.0), Point::new(-1.0, 2.0), Point::new(-1.0, 1.0)];
        let shape2 = vec![Point::new(1.0, 1.0), Point::new(-3.0, 2.0), Point::new(-1.0, 3.0)];
        let shape3 = vec![Point::new(1.0, 2.0), Point::new(-3.0, 3.0), Point::new(-1.0, 4.0)];

        let p1 = SimplePolygon::from_points(&shape1);
        let p2 = SimplePolygon::from_points(&shape2);
        let p3 = SimplePolygon::from_points(&shape3);

        assert!(p1.intersects(&p2));
        assert!(!p1.intersects(&p3));
    }

    #[test]
    fn convex_polygon_distance_test() {
        let shape1 = vec![Point::new(0.0, 0.0), Point::new(0.0, 1.0), Point::new(1.0, 1.0), Point::new(1.0, 0.0)];
        let shape2 = vec![Point::new(4.0, 5.0), Point::new(4.0, 6.0), Point::new(5.0, 6.0), Point::new(5.0, 5.0)];

        let p1 = SimplePolygon::from_points(&shape1);
        let p2 = SimplePolygon::from_points(&shape2);

        let cp1 = p1.convex_hull();
        let cp2 = p2.convex_hull();

        let dist = cp1.distance(&cp2);

        dbg!(dist);

        assert!((dist - 5.0).abs() < EPS);
    }
}