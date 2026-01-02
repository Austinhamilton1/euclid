use criterion::{criterion_group, criterion_main, Criterion};

use euclid::geometry::{polygon::{ConvexPolygon, SimplePolygon}, primitives::*};
use rand::Rng;

fn generate_simple_polygon(delta: Point) -> SimplePolygon{
    let points = vec![
        Point::new(-1.7, -5.0) + delta,
        Point::new(-2.9, -2.4) + delta,
        Point::new(-4.0, -4.5) + delta,
        Point::new(-4.5, 0.5) + delta,
        Point::new(-3.5, 0.5) + delta,
        Point::new(-3.5, 2.0) + delta,
        Point::new(-2.0, 1.0) + delta,
        Point::new(1.9, 5.0) + delta,
        Point::new(3.1, 3.0) + delta,
        Point::new(4.6, 3.1) + delta,
        Point::new(4.5, -1.0) + delta,
        Point::new(3.0, -0.9) + delta,
        Point::new(1.5, -4.9) + delta,
        Point::new(0.0, -3.0) + delta,
    ];

    SimplePolygon::from_points(&points)
}

fn generate_convex_polygon(delta: Point) -> ConvexPolygon {
    let points = vec![
        Point::new(-1.7, -5.0) + delta,
        Point::new(-2.9, -2.4) + delta,
        Point::new(-4.0, -4.5) + delta,
        Point::new(-4.5, 0.5) + delta,
        Point::new(-3.5, 0.5) + delta,
        Point::new(-3.5, 2.0) + delta,
        Point::new(-2.0, 1.0) + delta,
        Point::new(1.9, 5.0) + delta,
        Point::new(3.1, 3.0) + delta,
        Point::new(4.6, 3.1) + delta,
        Point::new(4.5, -1.0) + delta,
        Point::new(3.0, -0.9) + delta,
        Point::new(1.5, -4.9) + delta,
        Point::new(0.0, -3.0) + delta,
    ];

    let p = SimplePolygon::from_points(&points);

    p.convex_hull()
}

fn criterion_benchmark(c: &mut Criterion) {
    let mut rng = rand::thread_rng();

    // Generate random segments for benchmarking
    let mut segments: Vec<Segment> = Vec::new();
    for _ in 0..1500 {
        let x1 = rng.gen_range(-100.0..=100.0);
        let y1 = rng.gen_range(-100.0..=100.0);
        let x2 = rng.gen_range(-100.0..=100.0);
        let y2 = rng.gen_range(-100.0..=100.0);

        let p1 = Point::new(x1, y1);
        let p2 = Point::new(x2, y2);

        let s = Segment::new(p1, p2);

        segments.push(s);
    }

    // Generate random simple polygons for benchmarking
    let mut simple_polygons: Vec<SimplePolygon> = Vec::new();
    for _ in 0..1500 {
        let delta_x = rng.gen_range(-100.0..=100.0);
        let delta_y = rng.gen_range(-100.0..=100.0);
        let delta = Point::new(delta_x, delta_y);

        let p = generate_simple_polygon(delta);
        simple_polygons.push(p);
    }

    // Generate random convex polygons for benchmarking
    let mut convex_polygons: Vec<ConvexPolygon> = Vec::new();
    for _ in 0..1500 {
        let delta_x = rng.gen_range(-100.0..=100.0);
        let delta_y = rng.gen_range(-100.0..=100.0);
        let delta = Point::new(delta_x, delta_y);

        let p = generate_convex_polygon(delta);
        convex_polygons.push(p);
    }

    c.bench_function(
        "1500 Segments All Intersections",
        |b| {
            b.iter(|| {
                for i in 0..segments.len() - 1 {
                    for j in i+1..segments.len() {
                        segments[i].intersects(&segments[j]);
                    }
                }
            });
        }
    );

    c.bench_function(
        "1500 Simple Polygons All Intersections", 
        |b| {
            b.iter(|| {
                for i in 0..simple_polygons.len() - 1 {
                    for j in i+1..simple_polygons.len() {
                        simple_polygons[i].intersects(&simple_polygons[j]);
                    }
                }
            });
        }
    );

    c.bench_function(
        "1500 Convex Polygons All Intersections", 
        |b| {
            b.iter(|| {
                for i in 0..convex_polygons.len() - 1 {
                    for j in i+1..convex_polygons.len() {
                        convex_polygons[i].intersects(&convex_polygons[j]);
                    }
                }
            });
        }
    );
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);