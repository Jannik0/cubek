use cubecl::comptime;
use cubecl::cube;
use cubecl::prelude::*;
use serde::{Deserialize, Serialize};

use crate::components::instructions::AccumulatorFormat;

use crate::components::instructions::max_identity;
use crate::components::instructions::plane_topk_insert;
use crate::components::instructions::plane_topk_merge;
use crate::components::instructions::reaches;
use crate::components::instructions::{Accumulator, Item, Value, ValueExpand};
use crate::components::instructions::{numeric_is_nan, topk_walk_values, topk_walk_with_coords};
use crate::{
    ReduceFamily, ReduceInstruction, ReducePrecision,
    components::instructions::{
        ReduceOutputMode, ReduceRequirements, ReduceStep, ReduceWithIndices,
        ReduceWithIndicesFamily, SharedAccumulator,
    },
};
use cubecl::frontend::Numeric;

#[derive_cube_comptime]
#[derive(Serialize, Deserialize)]
pub struct TopKConfig {
    pub k: usize,
    pub output: ReduceOutputMode,
}

#[derive(Debug, CubeType, Clone)]
pub struct TopK {
    #[cube(comptime)]
    pub k: usize,
    #[cube(comptime)]
    pub output: ReduceOutputMode,
}

impl ReduceFamily for TopK {
    type Instruction<P: ReducePrecision> = Self;
    type Config = TopKConfig;
}

impl ReduceWithIndicesFamily for TopK {
    type Instruction<P: ReducePrecision> = Self;
    type Config = TopKConfig;
}

/// Insert `insert_val` into the descending-sorted `elements` (and its
/// coordinate, when it carries one), pushing the smallest slot out.
///
/// Ties break towards the lower coordinate, matching the CPU reference. A
/// coordinate-less candidate emits no index arithmetic at all.
///
/// A candidate that reaches no component's last kept slot changes nothing and skips
/// the `k`-slot walk — over a long row, almost every candidate.
#[cube]
pub(crate) fn topk_insert<N: Numeric, S: Size>(
    elements: &mut Array<Vector<N, S>>,
    coordinates: &mut Value<Vector<u32, S>>,
    insert_val: Vector<N, S>,
    insert_coord: &Value<Vector<u32, S>>,
    #[comptime] k: usize,
) {
    if reaches(insert_val, elements[k - 1]) {
        match insert_coord {
            Value::None => topk_walk_values(elements, insert_val, k, false),
            Value::Single(coord) => topk_walk_with_coords(
                elements,
                coordinates.multiple_mut(),
                insert_val,
                coord.unwrap(),
                k,
                false,
            ),
            Value::Multiple(_) => panic!("a top-k candidate carries at most one coordinate"),
        }
    }
}

#[derive(CubeType)]
pub struct TopKSharedAccumulator<P: ReducePrecision> {
    elements: Sequence<Shared<[Vector<P::EA, P::SI>]>>,
    /// Empty unless the instruction tracks coordinates; its length is the single
    /// source of truth for whether coordinates are staged (see `read`/`write`).
    args: Sequence<Shared<[Vector<u32, P::SI>]>>,
    #[cube(comptime)]
    k: usize,
}

#[cube]
impl<P: ReducePrecision> SharedAccumulator<P, TopK> for TopKSharedAccumulator<P> {
    fn allocate(#[comptime] length: usize, #[comptime] _coordinate: bool, inst: &TopK) -> Self {
        let has_coords = comptime!(inst.output.has_indices());

        // Both loops must be unrolled: a `Sequence` is built at expand time, so a
        // runtime loop would run the body once and leave a single slice behind
        // whatever `k` is, and `read`/`write` would then index past the end.
        let mut elements = Sequence::new();
        #[unroll]
        for _ in 0..inst.k {
            elements.push(Shared::new_slice(length));
        }

        let mut args = Sequence::new();
        if has_coords {
            #[unroll]
            for _ in 0..inst.k {
                args.push(Shared::new_slice(length));
            }
        }

        TopKSharedAccumulator::<P> {
            elements,
            args,
            k: inst.k,
        }
    }

    fn read(accumulator: &Self, index: usize) -> Accumulator<P> {
        let mut values = Array::new(accumulator.k);
        #[unroll]
        for i in 0..accumulator.k {
            values[i] = accumulator.elements[i][index];
        }

        let num_args = comptime!(accumulator.args.len());
        let args = if num_args != 0 {
            let mut args = Array::new(accumulator.k);
            #[unroll]
            for i in 0..accumulator.k {
                args[i] = accumulator.args[i][index];
            }
            Value::new_Multiple(args)
        } else {
            Value::new_None()
        };

        Accumulator::<P> {
            elements: Value::new_Multiple(values),
            args,
        }
    }

    fn write(accumulator: &mut Self, index: usize, item: Accumulator<P>) {
        let values = item.elements.multiple();
        #[unroll]
        for i in 0..accumulator.k {
            let acc = values[i];
            let shared_acc = &mut accumulator.elements[i];
            shared_acc[index] = acc;
        }

        let num_args = comptime!(accumulator.args.len());
        if num_args != 0 {
            let args = item.args.multiple();
            #[unroll]
            for i in 0..accumulator.k {
                let arg = args[i];
                let shared_arg_acc = &mut accumulator.args[i];
                shared_arg_acc[index] = arg;
            }
        }
    }
}

#[cube]
impl<P: ReducePrecision> ReduceInstruction<P> for TopK {
    type SharedAccumulator = TopKSharedAccumulator<P>;
    type Config = TopKConfig;

    fn requirements(this: &Self) -> super::ReduceRequirements {
        ReduceRequirements {
            coordinates: comptime!(this.output.has_indices()),
        }
    }

    fn accumulator_format(this: &Self) -> comptime_type!(AccumulatorFormat) {
        comptime!(AccumulatorFormat::Multiple(this.k))
    }

    fn from_config(#[comptime] config: Self::Config) -> Self {
        TopK {
            k: config.k,
            output: config.output,
        }
    }

    fn null_input(_this: &Self) -> Vector<P::EI, P::SI> {
        Vector::empty().fill(max_identity::<P::EI>())
    }

    fn null_accumulator(this: &Self) -> Accumulator<P> {
        let mut elements = Array::new(comptime!(this.k));
        #[unroll]
        for i in 0..this.k {
            elements[i] = Vector::new(max_identity::<P::EA>());
        }

        let args = if comptime!(this.output.has_indices()) {
            let mut args = Array::new(comptime!(this.k));
            #[unroll]
            for i in 0..this.k {
                args[i] = Vector::new(u32::MAX);
            }
            Value::new_Multiple(args)
        } else {
            Value::new_None()
        };

        Accumulator::<P> {
            elements: Value::new_Multiple(elements),
            args,
        }
    }

    fn reduce(
        this: &Self,
        accumulator: &mut Accumulator<P>,
        item: Item<P>,
        #[comptime] reduce_step: ReduceStep,
    ) {
        let elements = accumulator.elements.multiple_mut();

        match reduce_step {
            ReduceStep::Plane => {
                plane_topk_insert::<P::EA, P::SI>(
                    elements,
                    &mut accumulator.args,
                    Vector::cast_from(item.elements),
                    &item.args,
                    this.k,
                );
            }
            ReduceStep::Identity => {
                topk_insert::<P::EA, P::SI>(
                    elements,
                    &mut accumulator.args,
                    Vector::cast_from(item.elements),
                    &item.args,
                    this.k,
                );
            }
        }
    }

    fn plane_reduce_inplace(this: &Self, accumulator: &mut Accumulator<P>) {
        plane_topk_merge::<P::EA, P::SI>(
            accumulator.elements.multiple_mut(),
            &mut accumulator.args,
            this.k,
        );
    }

    fn fuse_accumulators(this: &Self, accumulator: &mut Accumulator<P>, other: &Accumulator<P>) {
        let elements = accumulator.elements.multiple_mut();
        let other_elements = other.elements.multiple();

        for i in 0..this.k {
            topk_insert::<P::EA, P::SI>(
                elements,
                &mut accumulator.args,
                other_elements[i],
                &other.args.slot(i),
                this.k,
            );
        }
    }

    fn output_mode(this: &Self) -> comptime_type!(ReduceOutputMode) {
        comptime!(this.output)
    }

    fn to_output_parallel<Out: Numeric, Idx: Numeric>(
        this: &Self,
        accumulator: Accumulator<P>,
        _shape_axis_reduce: usize,
    ) -> (Value<Out>, Value<Idx>) {
        match accumulator.args {
            Value::None => {
                let values = topk_finalize_values::<P, Out>(&accumulator, this.k);
                (Value::new_Multiple(values), Value::new_None())
            }
            Value::Multiple(_) => {
                let (values, coords) = topk_finalize_with_coords::<P>(&accumulator, this.k);

                let mut out_values = Array::new(this.k);
                let mut out_indices = Array::new(this.k);
                #[unroll]
                for i in 0..this.k {
                    out_values[i] = Out::cast_from(values[i]);
                    out_indices[i] = Idx::cast_from(coords[i]);
                }

                (
                    Value::new_Multiple(out_values),
                    Value::new_Multiple(out_indices),
                )
            }
            Value::Single(_) => panic!("top-k accumulator coordinates are one slice per slot"),
        }
    }

    fn to_output_perpendicular<Out: Numeric, Idx: Numeric>(
        this: &Self,
        accumulator: Accumulator<P>,
        _shape_axis_reduce: usize,
    ) -> (Value<Vector<Out, P::SI>>, Value<Vector<Idx, P::SI>>) {
        let acc_values = accumulator.elements.multiple();
        let mut out_values = Array::new(this.k);
        #[unroll]
        for i in 0..this.k {
            out_values[i] = Vector::cast_from(acc_values[i]);
        }

        let indices = match &accumulator.args {
            Value::None => Value::new_None(),
            Value::Multiple(acc_args) => {
                let mut out_indices = Array::new(this.k);
                #[unroll]
                for i in 0..this.k {
                    out_indices[i] = Vector::cast_from(acc_args[i]);
                }
                Value::new_Multiple(out_indices)
            }
            Value::Single(_) => panic!("top-k accumulator coordinates are one slice per slot"),
        };

        (Value::new_Multiple(out_values), indices)
    }
}

impl<P: ReducePrecision> ReduceWithIndices<P> for TopK {}

/// Collapse the `k * vector_size` accumulator candidates down to the final `k`
/// values, for the parallel (reduce axis is the vectorized axis) layout.
///
/// Coordinates are not tracked, so ties are broken arbitrarily. Use
/// [`topk_finalize_with_coords`] when indices are wanted.
#[cube]
fn topk_finalize_values<P: ReducePrecision, Out: Numeric>(
    accumulator: &Accumulator<P>,
    #[comptime] k: usize,
) -> Array<Out> {
    let vals = accumulator.elements.multiple();
    let vector_size = vals[0].vector_size().comptime();
    let unrolled =
        comptime!(k * k * vector_size <= crate::components::instructions::TOPK_UNROLL_BUDGET);

    let mut topk = Array::new(k);
    #[unroll]
    for slot in 0..k {
        topk[slot] = max_identity::<Out>();
    }

    #[unroll(unrolled)]
    for i in 0..k {
        #[unroll]
        for j in 0..vector_size {
            let mut element = Out::cast_from(vals[i].extract(j));

            if scalar_is_nan::<Out>(element) {
                // Rare, and a loop (see `topk_walk_values`): a NaN passes the NaN slots only.
                if unrolled {
                    for _step in 0..k {
                        let current = topk[0];
                        let keep = scalar_is_nan::<Out>(current);
                        let passed = select(keep, element, current);
                        #[unroll]
                        for slot in 0..k - 1 {
                            topk[slot] = topk[slot + 1];
                        }
                        topk[k - 1] = select(keep, current, element);
                        element = passed;
                    }
                } else {
                    for slot in 0..k {
                        let current = topk[slot];
                        let keep = scalar_is_nan::<Out>(current);
                        topk[slot] = select(keep, current, element);
                        element = select(keep, element, current);
                    }
                }
            } else {
                // A NaN slot compares false, so it keeps its place ahead.
                #[unroll(unrolled)]
                for slot in 0..k {
                    let current = topk[slot];
                    let takes = current < element;
                    topk[slot] = select(takes, element, current);
                    element = select(takes, current, element);
                }
            }
        }
    }

    topk
}

/// Collapse the `k * vector_size` accumulator candidates down to the final `k`
/// values *and* their coordinates, for the parallel layout.
///
/// Ties break towards the lower coordinate, matching the CPU reference. The
/// accumulator must have been built with coordinate tracking on.
#[cube]
fn topk_finalize_with_coords<P: ReducePrecision>(
    accumulator: &Accumulator<P>,
    #[comptime] k: usize,
) -> (Array<P::EA>, Array<u32>) {
    let vals = accumulator.elements.multiple();
    let coords = accumulator.args.multiple();
    let vector_size = coords[0].vector_size().comptime();
    let unrolled =
        comptime!(k * k * vector_size <= crate::components::instructions::TOPK_UNROLL_BUDGET);

    let mut topk_vals = Array::new(k);
    let mut topk_coords = Array::new(k);

    #[unroll]
    for slot in 0..k {
        topk_vals[slot] = max_identity::<P::EA>();
        topk_coords[slot] = u32::MAX;
    }

    #[unroll(unrolled)]
    for i in 0..k {
        #[unroll]
        for j in 0..vector_size {
            let mut value = vals[i].extract(j);
            let mut coordinate = coords[i].extract(j);

            if scalar_is_nan::<P::EA>(value) {
                // Rare, and a loop (see `topk_walk_with_coords`): a NaN passes the NaN
                // slots at a lower coordinate only.
                if unrolled {
                    for _step in 0..k {
                        let current_value = topk_vals[0];
                        let current_coordinate = topk_coords[0];
                        let keep = scalar_is_nan::<P::EA>(current_value)
                            && current_coordinate < coordinate;
                        let passed_value = select(keep, value, current_value);
                        let passed_coordinate = select(keep, coordinate, current_coordinate);
                        #[unroll]
                        for slot in 0..k - 1 {
                            topk_vals[slot] = topk_vals[slot + 1];
                            topk_coords[slot] = topk_coords[slot + 1];
                        }
                        topk_vals[k - 1] = select(keep, current_value, value);
                        topk_coords[k - 1] = select(keep, current_coordinate, coordinate);
                        value = passed_value;
                        coordinate = passed_coordinate;
                    }
                } else {
                    for slot in 0..k {
                        let current_value = topk_vals[slot];
                        let current_coordinate = topk_coords[slot];
                        let keep = scalar_is_nan::<P::EA>(current_value)
                            && current_coordinate < coordinate;
                        topk_vals[slot] = select(keep, current_value, value);
                        topk_coords[slot] = select(keep, current_coordinate, coordinate);
                        value = select(keep, value, current_value);
                        coordinate = select(keep, coordinate, current_coordinate);
                    }
                }
            } else {
                // A NaN slot compares false, so it keeps its place ahead.
                #[unroll(unrolled)]
                for slot in 0..k {
                    let current_value = topk_vals[slot];
                    let current_coordinate = topk_coords[slot];
                    let takes = select(
                        current_value == value,
                        coordinate < current_coordinate,
                        current_value < value,
                    );
                    topk_vals[slot] = select(takes, value, current_value);
                    topk_coords[slot] = select(takes, coordinate, current_coordinate);
                    value = select(takes, current_value, value);
                    coordinate = select(takes, current_coordinate, coordinate);
                }
            }
        }
    }

    (topk_vals, topk_coords)
}

/// Whether the scalar `value` is a NaN; never for an integer `E`.
#[cube]
fn scalar_is_nan<E: Numeric>(value: E) -> bool {
    let mut nan = false;
    let elem_type = elem_type_of::<E>();
    if comptime!(elem_type.is_float()) {
        nan = numeric_is_nan(Vector::<E, Const<1>>::new(value)).extract(0usize);
    }
    nan
}
