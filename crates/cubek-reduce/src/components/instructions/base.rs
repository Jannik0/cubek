use crate::components::{
    instructions::{max_identity, nan_value, numeric_is_nan},
    precision::ReducePrecision,
};
use cubecl::prelude::*;
use serde::{Deserialize, Serialize};

/// Which of a reduction's two results the single-output path writes.
///
/// For instructions that can track their candidates' coordinates
/// ([`TopK`](super::TopK), [`Min`](super::Min), [`Max`](super::Max)), the mode
/// only decides construction ([`ReduceInstruction::requirements`],
/// `null_accumulator`) and which half of the `to_output_*` pair the writer
/// keeps; everything in between reads the accumulator's own state.
///
/// The fused path (values *and* indices) is not a third variant here: it writes
/// both halves regardless, sizing the accumulator with [`Self::Indices`] so
/// coordinates are tracked.
#[derive_cube_comptime]
#[derive(Serialize, Deserialize)]
pub enum ReduceOutputMode {
    /// Write only the reduced values.
    Values,
    /// Write only the coordinates of the reduced values.
    Indices,
}

impl ReduceOutputMode {
    /// Whether coordinates must be tracked through the reduction.
    pub fn has_indices(&self) -> bool {
        matches!(self, ReduceOutputMode::Indices)
    }
}

pub trait ReduceFamily: Send + Sync + 'static + std::fmt::Debug {
    type Instruction<P: ReducePrecision>: ReduceInstruction<P, Config = Self::Config>;
    type Config: CubeComptime + Send + Sync;
}

/// A [`ReduceFamily`] whose instruction can emit values and indices together.
///
/// The bound lives on the trait rather than on a `where` clause at the kernel, because the
/// `#[cube(launch_unchecked)]` macro does not carry a where clause into the kernel struct it
/// generates. Implement this for any instruction that implements [`ReduceWithIndices`], and
/// `reduce_with_indices_kernel` works for it with no new kernel.
pub trait ReduceWithIndicesFamily: Send + Sync + 'static + std::fmt::Debug {
    type Instruction<P: ReducePrecision>: ReduceWithIndices<P, Config = Self::Config>;
    type Config: CubeComptime + Send + Sync;
}

#[derive(CubeType, Clone, Copy)]
#[expand(derive(Clone, Copy))]
/// Whether we keep track of coordinates of items
pub struct ReduceRequirements {
    #[cube(comptime)]
    pub coordinates: bool,
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq, CubeType)]
pub enum AccumulatorFormat {
    Multiple(usize),
    Single,
}

impl AccumulatorFormat {
    pub fn len(&self) -> usize {
        match self {
            AccumulatorFormat::Multiple(k) => *k,
            AccumulatorFormat::Single => 1,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[derive(CubeType)]
/// Whether the accumulator has zero, one or more vectors
pub enum Value<X: CubePrimitive> {
    Multiple(Array<X>),
    /// Wrap the item to be able to modify it as a field
    Single(ValueWrapper<X>),
    None,
}

#[derive(CubeType)]
/// Wrap the item to be able to modify it as a field
pub struct ValueWrapper<X: CubePrimitive> {
    val: X,
}

#[cube]
impl<X: CubePrimitive> ValueWrapper<X> {
    pub fn unwrap(&self) -> X {
        self.val
    }
}

#[cube]
impl<X: CubePrimitive> Value<X> {
    pub fn new_single(val: X) -> Value<X> {
        Value::new_Single(ValueWrapper::<X> { val })
    }

    pub fn item(&self) -> X {
        match self {
            Value::Multiple(_) => panic!("Tried item on Multiple"),
            Value::Single(item) => item.val,
            Value::None => panic!("Tried item on None"),
        }
    }

    pub fn multiple(&self) -> &Array<X> {
        match self {
            Value::Multiple(array) => array,
            Value::Single(_) => panic!("Tried multiple on Single"),
            Value::None => panic!("Tried multiple on None"),
        }
    }

    pub fn multiple_mut(&mut self) -> &mut Array<X> {
        match self {
            Value::Multiple(array) => array,
            Value::Single(_) => panic!("Tried multiple on Single"),
            Value::None => panic!("Tried multiple on None"),
        }
    }

    pub fn assign(&mut self, other: &Value<X>) {
        match (self, other) {
            (Value::Multiple(this), Value::Multiple(other)) => {
                for i in 0..this.len() {
                    this[i] = other[i];
                }
            }
            (Value::Single(this), Value::Single(other)) => {
                this.val = other.val;
            }
            (Value::None, Value::None) => {}
            _ => panic!("Tried assigning different accumulator kinds"),
        }
    }

    /// The `index`-th candidate as a standalone value; `None` stays `None`.
    pub fn slot(&self, index: usize) -> Value<X> {
        match self {
            Value::Multiple(array) => Value::new_single(array[index]),
            Value::Single(item) => Value::new_single(item.val),
            Value::None => Value::new_None(),
        }
    }
}

/// How much fully-unrolled top-k selection work is worth emitting, counted in
/// copies of a loop body.
///
/// The selection networks below are `k`-by-`k`: every candidate walks all `k`
/// accumulator slots. Unrolling both levels keeps the accumulator in registers
/// with constant slot indices, which is worth a lot — rolled, the slots move to
/// scratch memory and every step becomes a load/store. Measured on a
/// `32x512x4095` `ArgTopK` (device timing, RTX 5090 / Vulkan), unrolling is
/// worth 5.8x at `k = 32` and 2.2x at `k = 64`.
///
/// But the emitted kernel grows with the product, so a flat cap on `k` prices
/// the three nest shapes wrong: `topk_finalize_*` is `k * k * vector_size`, not
/// `k * k`, and is the first to become unaffordable. Budgeting the product
/// instead lets the square nests unroll further than the cubic one, and stops
/// all of them before the backend compiler does: past this budget the same
/// selection runs as a plain runtime loop whose kernel size does not depend on
/// `k` at all.
pub(crate) const TOPK_UNROLL_BUDGET: usize = 1024;

/// Whether any component of `item` reaches the list slot `kth` — reaches, not only
/// beats, so a tie still goes through the insertion and its coordinate rule.
///
/// The negation is load-bearing and is not `>=`: this guard only skips what
/// provably cannot enter, so a NaN, which compares unordered against every
/// slot, has to reach the insertion the way it did before the guard existed.
/// There it takes slot 0 — `elements[j] > NaN` is false — and carries its own
/// coordinate out. Under `>=` a NaN would fail the guard instead, and an
/// all-NaN row would insert nothing and emit the `u32::MAX` null-accumulator
/// sentinel as its index.
#[cube]
#[allow(clippy::neg_cmp_op_on_partial_ord)]
pub(crate) fn reaches<N: Numeric, S: Size>(item: Vector<N, S>, kth: Vector<N, S>) -> bool {
    let mut any = false;
    #[unroll]
    for i in 0..item.vector_size().comptime() {
        if !(item.extract(i) < kth.extract(i)) {
            any = true;
        }
    }
    any
}

/// Where `lhs` is ahead of `rhs` in top-k order by value alone: a NaN is ahead of every
/// number, as in `torch.topk`, then the larger number. Two NaNs, and equal numbers, are
/// [`topk_same`].
#[cube]
pub(crate) fn topk_ahead<N: Numeric, S: Size>(
    lhs: Vector<N, S>,
    rhs: Vector<N, S>,
) -> Vector<bool, S> {
    let by_order = lhs.greater_than(&rhs);
    let elem_type = elem_type_of::<N>();
    if comptime!(elem_type.is_float()) {
        let lhs_nan = numeric_is_nan(lhs);
        let rhs_nan = numeric_is_nan(rhs);
        by_order.or(lhs_nan.vec_and(!rhs_nan))
    } else {
        by_order
    }
}

/// Where `lhs` and `rhs` tie by value in top-k order: equal, or both NaN.
#[cube]
pub(crate) fn topk_same<N: Numeric, S: Size>(
    lhs: Vector<N, S>,
    rhs: Vector<N, S>,
) -> Vector<bool, S> {
    let equal = lhs.equal(&rhs);
    let elem_type = elem_type_of::<N>();
    if comptime!(elem_type.is_float()) {
        equal.or(numeric_is_nan(lhs).vec_and(numeric_is_nan(rhs)))
    } else {
        equal
    }
}

/// Where the slot holding `kept` at `kept_coord` stays ahead of `candidate` at `coord`:
/// ahead by value, or tied by value at a lower coordinate.
#[cube]
pub(crate) fn topk_keeps<N: Numeric, S: Size>(
    kept: Vector<N, S>,
    kept_coord: Vector<u32, S>,
    candidate: Vector<N, S>,
    coord: Vector<u32, S>,
) -> Vector<bool, S> {
    select_many(
        topk_same(kept, candidate),
        kept_coord.less_than(&coord),
        topk_ahead(kept, candidate),
    )
}

/// Whether `value` holds a NaN on any lane; never for an integer `N`.
#[cube]
fn any_nan<N: Numeric, S: Size>(value: Vector<N, S>) -> bool {
    let mut any = false;
    let elem_type = elem_type_of::<N>();
    if comptime!(elem_type.is_float()) {
        let nan = numeric_is_nan(value);
        #[unroll]
        for i in 0..value.vector_size().comptime() {
            if nan.extract(i) {
                any = true;
            }
        }
    }
    any
}

/// `value` walked down the sorted list `elements`, which keeps the `k` best: each slot keeps
/// the better of what it holds and what comes down, and passes the other on.
///
/// A value that is no NaN on any lane, almost every value, walks with the plain compare: it
/// takes a slot it beats. A NaN slot compares false, so it stays ahead, as the order wants.
/// A value with a NaN, rare, walks in the full order ([`topk_ahead`]), and that walk stays a
/// loop: unrolled beside the first, it would double the code of every walk. For a list in
/// registers (`unrolled`), the loop turns the list instead of picking a slot at run time,
/// which would move the list to memory for the whole kernel.
#[cube]
pub(crate) fn topk_walk_values<N: Numeric, S: Size>(
    elements: &mut Array<Vector<N, S>>,
    value: Vector<N, S>,
    #[comptime] k: usize,
    #[comptime] unrolled: bool,
) {
    let mut value = value;
    if any_nan(value) {
        if unrolled {
            for _step in 0..k {
                let kept = elements[0];
                let keep = !topk_ahead(value, kept);
                let passed = select_many(keep, value, kept);
                #[unroll]
                for j in 0..k - 1 {
                    elements[j] = elements[j + 1];
                }
                elements[k - 1] = select_many(keep, kept, value);
                value = passed;
            }
        } else {
            for j in 0..k {
                let kept = elements[j];
                let keep = !topk_ahead(value, kept);
                elements[j] = select_many(keep, kept, value);
                value = select_many(keep, value, kept);
            }
        }
    } else {
        #[unroll(unrolled)]
        for j in 0..k {
            let kept = elements[j];
            let takes = kept.less_than(&value);
            elements[j] = select_many(takes, value, kept);
            value = select_many(takes, kept, value);
        }
    }
}

/// [`topk_walk_values`] with each value's coordinate riding along: a tie by value goes to
/// the lower coordinate ([`topk_keeps`]).
#[cube]
pub(crate) fn topk_walk_with_coords<N: Numeric, S: Size>(
    elements: &mut Array<Vector<N, S>>,
    coordinates: &mut Array<Vector<u32, S>>,
    value: Vector<N, S>,
    coordinate: Vector<u32, S>,
    #[comptime] k: usize,
    #[comptime] unrolled: bool,
) {
    let mut value = value;
    let mut coordinate = coordinate;
    if any_nan(value) {
        if unrolled {
            for _step in 0..k {
                let kept = elements[0];
                let kept_coordinate = coordinates[0];
                let keep = topk_keeps(kept, kept_coordinate, value, coordinate);
                let passed = select_many(keep, value, kept);
                let passed_coordinate = select_many(keep, coordinate, kept_coordinate);
                #[unroll]
                for j in 0..k - 1 {
                    elements[j] = elements[j + 1];
                    coordinates[j] = coordinates[j + 1];
                }
                elements[k - 1] = select_many(keep, kept, value);
                coordinates[k - 1] = select_many(keep, kept_coordinate, coordinate);
                value = passed;
                coordinate = passed_coordinate;
            }
        } else {
            for j in 0..k {
                let kept = elements[j];
                let kept_coordinate = coordinates[j];
                let keep = topk_keeps(kept, kept_coordinate, value, coordinate);
                elements[j] = select_many(keep, kept, value);
                coordinates[j] = select_many(keep, kept_coordinate, coordinate);
                value = select_many(keep, value, kept);
                coordinate = select_many(keep, coordinate, kept_coordinate);
            }
        }
    } else {
        #[unroll(unrolled)]
        for j in 0..k {
            let kept = elements[j];
            let kept_coordinate = coordinates[j];
            let takes = select_many(
                kept.equal(&value),
                coordinate.less_than(&kept_coordinate),
                kept.less_than(&value),
            );
            elements[j] = select_many(takes, value, kept);
            coordinates[j] = select_many(takes, coordinate, kept_coordinate);
            value = select_many(takes, kept, value);
            coordinate = select_many(takes, kept_coordinate, coordinate);
        }
    }
}

/// The plane's best `item` per lane in top-k order: a NaN where any unit holds one, the
/// largest number otherwise. `plane_max` alone drops a NaN on most targets.
#[cube]
fn plane_max_nan_first<N: Numeric, S: Size>(item: Vector<N, S>) -> Vector<N, S> {
    let max = plane_max(item);
    let elem_type = elem_type_of::<N>();
    if comptime!(elem_type.is_float()) {
        let any_nan = plane_max(Vector::<u32, S>::cast_from(numeric_is_nan(item)));
        select_many(
            any_nan.equal(&Vector::new(1u32)),
            Vector::new(nan_value::<N>()),
            max,
        )
    } else {
        max
    }
}

/// The lowest coordinate, per lane, among the plane's units whose `item` ties `target` in
/// top-k order ([`topk_same`]).
#[cube]
fn lowest_coordinate_tying<N: Numeric, S: Size>(
    target: Vector<N, S>,
    item: Vector<N, S>,
    coordinate: Vector<u32, S>,
) -> Vector<u32, S> {
    let candidate = select_many(
        topk_same(item, target),
        coordinate,
        Vector::empty().fill(u32::MAX),
    );
    plane_min(candidate)
}

/// Plane-cooperative top-k insertion; the candidate's coordinate decides which
/// algorithm runs, since winners are identified by their coordinate when one
/// rides along and by unit id otherwise.
///
/// A step none of whose components reaches the list's last kept slot changes nothing
/// and is skipped: the insertion is `k` plane reductions per step, and over a
/// long row almost every step is such a step — on a 151936-wide row of logits
/// the insertion was the whole cost of a top-20 (4.7 ms on GP100).
#[cube]
pub fn plane_topk_insert<N: Numeric, S: Size>(
    elements: &mut Array<Vector<N, S>>,
    coordinates: &mut Value<Vector<u32, S>>,
    item: Vector<N, S>,
    coord: &Value<Vector<u32, S>>,
    #[comptime] k: usize,
) {
    if plane_any(reaches(item, elements[k - 1])) {
        match coord {
            Value::None => plane_topk_insert_values(elements, item, k),
            Value::Single(coord) => plane_topk_insert_with_coords(
                elements,
                coordinates.multiple_mut(),
                item,
                coord.unwrap(),
                k,
            ),
            Value::Multiple(_) => panic!("a top-k candidate carries at most one coordinate"),
        }
    }
}

#[cube]
fn plane_topk_insert_with_coords<N: Numeric, S: Size>(
    elements: &mut Array<Vector<N, S>>,
    coordinates: &mut Array<Vector<u32, S>>,
    item: Vector<N, S>,
    coord: Vector<u32, S>,
    #[comptime] k: usize,
) {
    let mut local_best_val = item;
    let mut local_best_coord = coord;

    #[unroll(k * k <= crate::components::instructions::TOPK_UNROLL_BUDGET)]
    for _i in 0..k {
        let winning_val = plane_max_nan_first(local_best_val);
        let winning_coord = lowest_coordinate_tying(winning_val, local_best_val, local_best_coord);

        topk_walk_with_coords(
            elements,
            coordinates,
            winning_val,
            winning_coord,
            k,
            comptime!(k * k <= crate::components::instructions::TOPK_UNROLL_BUDGET),
        );

        // Winner masking logic
        let is_winner =
            topk_same(local_best_val, winning_val).vec_and(local_best_coord.equal(&winning_coord));
        local_best_val = select_many(is_winner, Vector::new(max_identity::<N>()), local_best_val);
        local_best_coord = select_many(is_winner, Vector::new(u32::MAX), local_best_coord);
    }
}

#[cube]
fn plane_topk_insert_values<N: Numeric, S: Size>(
    elements: &mut Array<Vector<N, S>>,
    item: Vector<N, S>,
    #[comptime] k: usize,
) {
    let mut local_best_val = item;
    let unit_id = Vector::new(UNIT_POS_X);

    #[unroll(k * k <= crate::components::instructions::TOPK_UNROLL_BUDGET)]
    for _i in 0..k {
        let winning_val = plane_max_nan_first(local_best_val);
        let is_match = topk_same(local_best_val, winning_val);
        let winning_unit = plane_min(select_many(is_match, unit_id, Vector::new(u32::MAX)));

        topk_walk_values(
            elements,
            winning_val,
            k,
            comptime!(k * k <= crate::components::instructions::TOPK_UNROLL_BUDGET),
        );

        // Winner masking logic
        let is_winner = unit_id.equal(&winning_unit);
        local_best_val = select_many(is_winner, Vector::new(max_identity::<N>()), local_best_val);
    }
}

/// Plane-cooperative merge of per-unit top-k candidates; the accumulator's
/// coordinates decide which algorithm runs, as in [`plane_topk_insert`].
#[cube]
pub fn plane_topk_merge<N: Numeric, S: Size>(
    elements: &mut Array<Vector<N, S>>,
    coordinates: &mut Value<Vector<u32, S>>,
    #[comptime] k: usize,
) {
    match coordinates {
        Value::None => plane_topk_merge_values(elements, k),
        Value::Multiple(coordinates) => plane_topk_merge_with_coords(elements, coordinates, k),
        Value::Single(_) => panic!("top-k accumulator coordinates are one slice per slot"),
    }
}

#[cube]
fn plane_topk_merge_with_coords<N: Numeric, S: Size>(
    elements: &mut Array<Vector<N, S>>,
    coordinates: &mut Array<Vector<u32, S>>,
    #[comptime] k: usize,
) {
    let mut final_elements = Array::new(k);
    let mut final_coords = Array::new(k);
    let mut cursor = Vector::new(0u32);
    let unit_id = Vector::new(UNIT_POS_X);

    #[unroll(k * k <= crate::components::instructions::TOPK_UNROLL_BUDGET)]
    for i in 0..k {
        let mut local_val = Vector::new(max_identity::<N>());
        let mut local_coord = Vector::new(u32::MAX);

        #[unroll(k * k <= crate::components::instructions::TOPK_UNROLL_BUDGET)]
        for j in 0..k {
            let is_pointed = cursor.equal(&Vector::new(j as u32));
            local_val = select_many(is_pointed, elements[j], local_val);
            local_coord = select_many(is_pointed, coordinates[j], local_coord);
        }

        let winning_val = plane_max_nan_first(local_val);
        let best_c = lowest_coordinate_tying(winning_val, local_val, local_coord);
        final_coords[i] = best_c;
        let is_cand = topk_same(local_val, winning_val).vec_and(local_coord.equal(&best_c));
        let winning_unit = plane_min(select_many(is_cand, unit_id, Vector::new(u32::MAX)));

        final_elements[i] = winning_val;
        let is_winner_thread = unit_id.equal(&winning_unit);
        cursor = select_many(is_winner_thread, cursor + Vector::new(1u32), cursor);
    }

    #[unroll]
    for i in 0..k {
        elements[i] = final_elements[i];
        coordinates[i] = final_coords[i];
    }
}

#[cube]
fn plane_topk_merge_values<N: Numeric, S: Size>(
    elements: &mut Array<Vector<N, S>>,
    #[comptime] k: usize,
) {
    let mut final_elements = Array::new(k);
    let mut cursor = Vector::new(0u32);
    let unit_id = Vector::new(UNIT_POS_X);

    #[unroll(k * k <= crate::components::instructions::TOPK_UNROLL_BUDGET)]
    for i in 0..k {
        let mut local_val = Vector::new(max_identity::<N>());

        #[unroll(k * k <= crate::components::instructions::TOPK_UNROLL_BUDGET)]
        for j in 0..k {
            let is_pointed = cursor.equal(&Vector::new(j as u32));
            local_val = select_many(is_pointed, elements[j], local_val);
        }

        let winning_val = plane_max_nan_first(local_val);
        let is_cand = topk_same(local_val, winning_val);
        let winning_unit = plane_min(select_many(is_cand, unit_id, Vector::new(u32::MAX)));

        final_elements[i] = winning_val;
        let is_winner_thread = unit_id.equal(&winning_unit);
        cursor = select_many(is_winner_thread, cursor + Vector::new(1u32), cursor);
    }

    #[unroll]
    for i in 0..k {
        elements[i] = final_elements[i];
    }
}

#[derive(CubeType)]
/// Whether the accumulator has zero, one or more vectors
/// This should be the same variant as AccumulatorKind for an instruction
pub enum SharedAccumulatorKind<X: CubePrimitive> {
    Multiple(Sequence<Shared<[X]>>),
    Single(Shared<[X]>),
    None,
}

#[cube]
impl<X: CubePrimitive> SharedAccumulatorKind<X> {
    pub fn get(&self, i: usize) -> Value<X> {
        match self {
            SharedAccumulatorKind::Multiple(sequence) => {
                let mut array = Array::new(sequence.len());
                #[unroll]
                for k_iter in 0..sequence.len() {
                    array[k_iter] = sequence[k_iter][i];
                }
                Value::new_Multiple(array)
            }
            SharedAccumulatorKind::Single(shared_memory) => Value::new_single(shared_memory[i]),
            SharedAccumulatorKind::None => Value::new_None(),
        }
    }

    pub fn set(&mut self, i: usize, value: Value<X>) {
        match self {
            SharedAccumulatorKind::Multiple(sequence) =>
            {
                #[unroll]
                for k_iter in 0..sequence.len() {
                    let shared_acc = &mut sequence[k_iter];
                    shared_acc[i] = value.multiple()[k_iter];
                }
            }
            SharedAccumulatorKind::Single(shared_memory) => shared_memory[i] = value.item(),
            SharedAccumulatorKind::None => {}
        }
    }
}

/// An instruction for a reduce algorithm that works with [`Vector`].
///
/// See a provided implementation, such as [`Sum`](super::Sum) or [`Max`](super::Max) for an example
/// how to implement this trait for a custom instruction.
///
/// A reduction works at three levels. First, it takes input data of type `In` and reduce them with
/// their coordinate into an `AccumulatorItem`. Then, multiple `AccumulatorItem` are possibly fused
/// together into a single accumulator that is converted to the expected output type.
#[cube]
pub trait ReduceInstruction<P: ReducePrecision>:
    Send + Sync + 'static + std::fmt::Debug + CubeType + Sized
{
    type Config: CubeComptime + Send + Sync;

    /// When multiple agents are collaborating to reduce a single slice, we need a share accumulator
    /// to store multiple `AccumulatorItem`. This is most likely a `Shared<[Vector<T>]>` or a struct
    /// or tuple of vectorized shared memories.
    type SharedAccumulator: SharedAccumulator<P, Self>;

    /// Requirements of the reduce.
    fn requirements(this: &Self) -> ReduceRequirements;
    fn accumulator_format(this: &Self) -> comptime_type!(AccumulatorFormat);

    fn from_config(#[comptime] config: Self::Config) -> Self;
    /// A input such that `Self::reduce(accumulator, Self::null_input(), coordinate, use_planes)`
    /// is guaranteed to return `accumulator` unchanged for any choice of `coordinate`.
    fn null_input(this: &Self) -> Vector<P::EI, P::SI>;

    /// A accumulator such that `Self::fuse_accumulators(accumulator, Self::null_accumulator()`
    /// always returns is guaranteed to return `accumulator` unchanged.
    fn null_accumulator(this: &Self) -> Accumulator<P>;

    /// If `ReduceStep` is `Plane`, reduce all the `item` and `coordinate` within the `accumulator`.
    /// if `ReduceStep` is `Identity`, reduce the given `item` and `coordinate` into the
    /// accumulator.
    fn reduce(
        this: &Self,
        accumulator: &mut Accumulator<P>,
        item: Item<P>,
        #[comptime] reduce_step: ReduceStep,
    );

    fn plane_reduce_inplace(this: &Self, accumulator: &mut Accumulator<P>);

    /// Reduce a whole accumulator (other) in accumulator.
    fn fuse_accumulators(this: &Self, accumulator: &mut Accumulator<P>, other: &Accumulator<P>);

    /// Which half of the `to_output_*` pair the single-output kernel writes.
    fn output_mode(this: &Self) -> comptime_type!(ReduceOutputMode);

    /// Reduce all elements of the accumulator into a single output element of type `Out`,
    /// with its coordinate as `Idx` when the accumulator tracks coordinates
    /// (`Value::None` otherwise).
    fn to_output_parallel<Out: Numeric, Idx: Numeric>(
        this: &Self,
        accumulator: Accumulator<P>,
        shape_axis_reduce: usize,
    ) -> (Value<Out>, Value<Idx>);

    /// Convert each element of the accumulator into the expected output element of type
    /// `Out`, with its coordinates as `Idx` when the accumulator tracks coordinates
    /// (`Value::None` otherwise).
    fn to_output_perpendicular<Out: Numeric, Idx: Numeric>(
        this: &Self,
        accumulator: Accumulator<P>,
        shape_axis_reduce: usize,
    ) -> (Value<Vector<Out, P::SI>>, Value<Vector<Idx, P::SI>>);
}

/// Marker for instructions whose `to_output_*` conversions emit a non-`None`
/// indices half whenever the accumulator tracks coordinates, so a fused reduce
/// can write both outputs from one launch.
///
/// A separate trait rather than a [`ReduceInstruction`] guarantee so that
/// instructions with no meaningful index (`Sum`, `Mean`, ...) are not accepted
/// by the fused entrypoint.
pub trait ReduceWithIndices<P: ReducePrecision>: ReduceInstruction<P> {}

#[derive(CubeType)]
pub struct Item<P: ReducePrecision> {
    pub elements: Vector<P::EI, P::SI>,
    // Warning: should not be Multiple
    pub args: Value<Vector<u32, P::SI>>,
}

#[derive(CubeType)]
pub struct Accumulator<P: ReducePrecision> {
    pub elements: Value<Vector<P::EA, P::SI>>,
    pub args: Value<Vector<u32, P::SI>>,
}

/// A simple trait that abstract over a single or multiple shared memory.
#[cube]
pub trait SharedAccumulator<P: ReducePrecision, I: ReduceInstruction<P>>:
    CubeType + 'static
{
    fn allocate(#[comptime] length: usize, #[comptime] _coordinate: bool, inst: &I) -> Self;

    fn read(accumulator: &Self, index: usize) -> Accumulator<P>;

    fn write(accumulator: &mut Self, index: usize, item: Accumulator<P>);
}

#[cube]
impl<P: ReducePrecision, I: ReduceInstruction<P>> SharedAccumulator<P, I>
    for Shared<[Vector<P::EA, P::SI>]>
{
    fn allocate(#[comptime] length: usize, #[comptime] _coordinate: bool, _inst: &I) -> Self {
        Shared::new_slice(length)
    }

    fn read(accumulator: &Self, index: usize) -> Accumulator<P> {
        Accumulator::<P> {
            elements: Value::new_single(accumulator[index]),
            args: Value::new_None(),
        }
    }

    fn write(accumulator: &mut Self, index: usize, item: Accumulator<P>) {
        accumulator[index] = item.elements.item();
    }
}

/// A pair of shared memory used for [`Max`](super::Max) and [`Min`](super::Min).
#[derive(CubeType)]
pub struct ArgAccumulator<P: ReducePrecision> {
    pub elements: Shared<[Vector<P::EA, P::SI>]>,
    /// Empty unless the instruction tracks coordinates; its length is the single
    /// source of truth for whether coordinates are staged (see `read`/`write`).
    pub args: Sequence<Shared<[Vector<u32, P::SI>]>>,
}

/// For a single reduce step whether we need to do plane reduction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReduceStep {
    /// Just keep the current value
    Identity,
    /// reduce across the plane
    Plane,
}

#[cube]
impl<P: ReducePrecision, I: ReduceInstruction<P>> SharedAccumulator<P, I> for ArgAccumulator<P> {
    fn allocate(#[comptime] length: usize, #[comptime] coordinate: bool, _inst: &I) -> Self {
        let mut args = Sequence::new();
        if coordinate {
            args.push(Shared::new_slice(length));
        }

        ArgAccumulator::<P> {
            elements: Shared::new_slice(length),
            args,
        }
    }

    fn read(accumulator: &Self, index: usize) -> Accumulator<P> {
        let num_args = comptime!(accumulator.args.len());
        let args = if num_args != 0 {
            Value::new_single(accumulator.args[0][index])
        } else {
            Value::new_None()
        };

        Accumulator::<P> {
            elements: Value::new_single(accumulator.elements[index]),
            args,
        }
    }

    fn write(accumulator: &mut Self, index: usize, item: Accumulator<P>) {
        accumulator.elements[index] = item.elements.item();

        let num_args = comptime!(accumulator.args.len());
        if num_args != 0 {
            let shared_args = &mut accumulator.args[0];
            shared_args[index] = item.args.item();
        }
    }
}

#[cube]
pub fn reduce_inplace<P: ReducePrecision, R: ReduceInstruction<P>>(
    inst: &R,
    accumulator: &mut Accumulator<P>,
    item: Item<P>,
    #[comptime] reduce_step: ReduceStep,
) {
    R::reduce(inst, accumulator, item, reduce_step)
}

#[cube]
pub fn reduce_shared_inplace<P: ReducePrecision, R: ReduceInstruction<P>>(
    inst: &R,
    accumulator: &mut R::SharedAccumulator,
    index: usize,
    item: Item<P>,
    #[comptime] reduce_step: ReduceStep,
) {
    let mut acc_item = R::SharedAccumulator::read(&*accumulator, index);
    R::reduce(inst, &mut acc_item, item, reduce_step);
    R::SharedAccumulator::write(accumulator, index, acc_item);
}

#[cube]
pub fn fuse_accumulator_inplace<P: ReducePrecision, R: ReduceInstruction<P>>(
    inst: &R,
    accumulator: &mut R::SharedAccumulator,
    destination: usize,
    origin: usize,
) {
    let mut acc = R::SharedAccumulator::read(&*accumulator, destination);
    R::fuse_accumulators(
        inst,
        &mut acc,
        &R::SharedAccumulator::read(&*accumulator, origin),
    );
    R::SharedAccumulator::write(accumulator, destination, acc);
}
