use super::*;

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
        for &point in self.vertices.iter() {
            center = center + point;
        }

        let len = self.vertices.len() as f64;

        center = (1.0 / len) * center;

        self.center = center;
    }

    pub fn convex_hull(&self) -> ConvexPolygon {
        assert!(self.vertices.len() >= 3);

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
        for &point in self.vertices.iter() {
            center = center + point;
        }

        let len = self.vertices.len() as f64;

        center = (1.0 / len) * center;

        self.center = center;
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

            for (i, &p) in self.vertices.iter().enumerate() {
                let r = self.vertices[(i + 1) % self.vertices.len()];
                let s = Segment::new(p, r);
                if test_ray.intersects(&s) {
                    intersect_count += 1;
                }
            }

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

/*
 * Handle a simplex from the GJK algorithm below.
 * Arguments:
 *     simplex: &mut Vec<Point> - Handle this simplex.
 *     dir: &mut Point - The direction vector for the support function.
 * Returns:
 *     bool - True if the simplex contains the origin, false otherwise.
 */
fn handle_simplex(simplex: &mut Vec<Point>, dir: &mut Point) -> bool {
    match simplex.len() {
        2 => handle_line(simplex, dir),
        3 => handle_triangle(simplex, dir),
        _ => unreachable!(),
    }
}

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
fn support(a: &impl Convex, b: &impl Convex, dir: Point) -> Point {
    let p1 = a.support(dir);
    let p2 = b.support(Point::new(-dir.x, -dir.y));
    p1 - p2
}

/*
 * Helper function for handle_simplex (line case).
 */
fn handle_line(simplex: &mut Vec<Point>, dir: &mut Point) -> bool {
    let a = simplex[1];
    let b = simplex[0];

    let ab = b - a;

    let ao = Point::new(-a.x, -a.y);

    *dir = triple_cross(ab, ao, ab);
    false
}

/*
 * Helper function for handle_simplex (triangle case).
 */
fn handle_triangle(simplex: &mut Vec<Point>, dir: &mut Point) -> bool {
    let a = simplex[2];
    let b = simplex[1];
    let c = simplex[0];

    let ab = b - a;
    let ac = c - a;
    let ao = Point::new(-a.x, -a.y);

    let ab_perp = triple_cross(ac, ab, ab);
    if ab_perp.dot(ao) > 0.0 {
        simplex.remove(0);
        *dir = ab_perp;
        return false;
    }

    let ac_perp = triple_cross(ab, ac, ac);
    if ac_perp.dot(ao) > 0.0 {
        simplex.remove(1);
        *dir = ac_perp;
        return false;
    }

    true
}

impl<T, U> Intersects<T> for U
    where 
        T: Convex,
        U: Convex,
    {
        fn intersects(&self, other: &T) -> bool {
            // Initial direction should be the vector of separation between
            // the centers of the shapes
            let mut dir = self.center() - other.center();
            if dir.x == 0.0 && dir.y == 0.0 {
                dir = Point::new(1.0, 0.0);
            }

            // Build the initial simplex
            let mut simplex: Vec<Point> = Vec::new();
            simplex.push(support(self, other, dir));
            dir = Point::new(-simplex[0].x, -simplex[0].y);

            // In a loop, successively build simplexes until one is found that contains
            // the origin (intersects) or one goes beyond the origin (does not intersect).
            loop {
                let p = support(self, other, dir);
                if p.dot(dir) <= 0.0 {
                    return false;
                }

                simplex.push(p);

                if handle_simplex(&mut simplex, &mut dir) {
                    return true;
                }
            }
        }
    }

impl Geometry for SimplePolygon {}
impl Geometry for ConvexPolygon {}

trait Shape2D {
    type Iter<'a>: Iterator<Item = &'a Point>
    where 
        Self: 'a;

    fn vertices(&self) -> Self::Iter<'_>;
}

impl Shape2D for SimplePolygon {
    type Iter<'a> = std::slice::Iter<'a, Point>;

    fn vertices(&self) -> Self::Iter<'_> {
        self.vertices.iter()
    }
}

impl Shape2D for ConvexPolygon {
    type Iter<'a> = std::slice::Iter<'a, Point>;

    fn vertices(&self) -> Self::Iter<'_> {
        self.vertices.iter()
    }
}

trait Convex {
    fn support(&self, dir: Point) -> Point;
    fn center(&self) -> Point;
}

impl Convex for ConvexPolygon {
    /*
     * Center function for grabbing the center of the Polygon.
     * Returns:
     *     Point - The center of mass.
     */
    fn center(&self) -> Point {
        self.center
    }

    /*
     * Support function for determining the maximal point in a specified direction.
     * Arguments:
     *     dir: Point
     * Returns:
     *     Point - The maximal point on the ConvexPolygon in the specified direction.
     */
    fn support(&self, dir: Point) -> Point {
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

        support_point
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-9;

    #[test]
    fn convex_hull_test() {
        let points = vec![Point::new(0.0, 0.0), Point::new(2.0, 0.0), Point::new(1.0, 1.0), Point::new(0.0, 2.0)];
        let sp = SimplePolygon::from_points(&points);
        let cp = sp.convex_hull();

        let mut iter = cp.vertices.iter();

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
}