//! Top-k over rows holding NaNs.
//!
//! `TopK` ranks a NaN ahead of every number, as `torch.topk` does: a row's NaNs come first, then
//! its largest numbers, every slot with an index inside the row. The rows hold none, one or two
//! NaNs at scattered coordinates, so a NaN arrives first, last and in between, in a vector lane
//! and in a plane's unit.

use cubecl::{
    config::autotune::AutotuneLevel,
    features::Plane,
    zspace::{Shape, Strides},
};
use cubek_reduce::{
    ReduceStrategy,
    launch::{RoutineStrategy, VectorizationStrategy},
    routines::{BlueprintStrategy, cube::CubeStrategy, plane::PlaneStrategy, unit::UnitStrategy},
};

use super::test_case::TestCase;

const K: usize = 4;

fn supports_plane() -> bool {
    let client = cubecl::test_device().client();
    client.properties().features.plane.contains(Plane::Ops)
}

fn strategy(routine: RoutineStrategy, parallel_output_vectorization: bool) -> ReduceStrategy {
    ReduceStrategy {
        autotune_level: AutotuneLevel::Full,
        vectorization: VectorizationStrategy {
            parallel_output_vectorization,
        },
        routine,
    }
}

fn unit() -> ReduceStrategy {
    strategy(
        RoutineStrategy::Unit(BlueprintStrategy::Inferred(UnitStrategy)),
        false,
    )
}

fn plane(independent: bool) -> ReduceStrategy {
    strategy(
        RoutineStrategy::Plane(BlueprintStrategy::Inferred(PlaneStrategy { independent })),
        false,
    )
}

fn cube(use_planes: bool) -> ReduceStrategy {
    strategy(
        RoutineStrategy::Cube(BlueprintStrategy::Inferred(CubeStrategy { use_planes })),
        false,
    )
}

fn run_topk(shape: Shape, strides: Strides, axis: usize, strategy: ReduceStrategy) {
    let case = TestCase::new::<f32>(shape, strides, Some(axis), strategy);
    let data = nan_data(&case.shape, axis);
    let case = case.with_data(data);
    case.test_topk(K);
    case.test_argtopk(K);
    case.test_topk_with_indices(K);
}

/// Distinct numbers, with slice `s` holding `s % 3` NaNs: none, one in the middle, or one at
/// each end of the row.
fn nan_data(shape: &Shape, axis: usize) -> Vec<f32> {
    let axis_len = shape[axis];
    let inner_len = shape.iter().skip(axis + 1).product::<usize>();
    let num_elements = shape.iter().product::<usize>();

    (0..num_elements)
        .map(|linear| {
            let at = (linear / inner_len) % axis_len;
            let slice = (linear / (inner_len * axis_len)) * inner_len + linear % inner_len;
            let nan = match slice % 3 {
                1 => at == axis_len / 2,
                2 => at == 0 || at + 1 == axis_len,
                _ => false,
            };
            match nan {
                true => f32::NAN,
                false => ((at * 37 + slice * 11) % 101) as f32 * 0.25 - 12.0,
            }
        })
        .collect()
}

#[test]
fn unit_parallel_f32() {
    run_topk(Shape::new([9, 12]), Strides::new(&[12, 1]), 1, unit());
}

#[test]
fn unit_perpendicular_f32() {
    run_topk(Shape::new([12, 9]), Strides::new(&[9, 1]), 0, unit());
}

#[test]
fn unit_short_row_f32() {
    run_topk(Shape::new([3, 8]), Strides::new(&[8, 1]), 1, unit());
}

#[test]
fn plane_parallel_f32() {
    if !supports_plane() {
        return;
    }
    for independent in [false, true] {
        run_topk(
            Shape::new([9, 50]),
            Strides::new(&[50, 1]),
            1,
            plane(independent),
        );
    }
}

#[test]
fn cube_parallel_f32() {
    for use_planes in [false, true] {
        if use_planes && !supports_plane() {
            continue;
        }
        run_topk(
            Shape::new([6, 300]),
            Strides::new(&[300, 1]),
            1,
            cube(use_planes),
        );
    }
}
