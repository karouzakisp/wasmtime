//! Elaboration phase: lowers EGraph back to sequences of operations
//! in CFG nodes.

use super::cost::Cost;
use super::OrderingInfo;
use super::Stats;
use super::VecDeque;
use crate::dominator_tree::DominatorTreePreorder;
use crate::hash_map::Entry as HashEntry;
use crate::inst_predicates::is_pure_for_egraph;
use crate::ir::{Block, Function, Inst, Value, ValueDef};
use crate::loop_analysis::{Loop, LoopAnalysis};
use crate::scoped_hash_map::ScopedHashMap;
use crate::trace;
use alloc::vec::Vec;
use cranelift_control::ControlPlane;
use cranelift_entity::{packed_option::ReservedValue, SecondaryMap};
use heapz::{DecreaseKey, Heap, RankPairingHeap};
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::{smallvec, SmallVec};

pub(crate) struct Elaborator<'a> {
    func: &'a mut Function,
    domtree: &'a DominatorTreePreorder,
    loop_analysis: &'a LoopAnalysis,
    /// Map from Value that is produced by a pure Inst (and was thus
    /// not in the side-effecting skeleton) to the value produced by
    /// an elaborated inst (placed in the layout) to whose results we
    /// refer in the final code.
    ///
    /// The first time we use some result of an instruction during
    /// elaboration, we can place it and insert an identity map (inst
    /// results to that same inst's results) in this scoped
    /// map. Within that block and its dom-tree children, that mapping
    /// is visible and we can continue to use it. This allows us to
    /// avoid cloning the instruction. However, if we pop that scope
    /// and use it somewhere else as well, we will need to
    /// duplicate. We detect this case by checking, when a value that
    /// we want is not present in this map, whether the producing inst
    /// is already placed in the Layout. If so, we duplicate, and
    /// insert non-identity mappings from the original inst's results
    /// to the cloned inst's results.
    ///
    /// Note that as values may refer to unions that represent a subset
    /// of a larger eclass, it's not valid to walk towards the root of a
    /// union tree: doing so would potentially equate values that fall
    /// on different branches of the dominator tree.
    value_to_elaborated_value: ScopedHashMap<Value, ElaboratedValue>,
    /// Map from Value to the best (lowest-cost) Value in its eclass
    /// (tree of union value-nodes).
    value_to_best_value: SecondaryMap<Value, BestEntry>,
    /// Stack of blocks and loops in current elaboration path.
    loop_stack: SmallVec<[LoopStackEntry; 8]>,
    /// The current block into which we are elaborating.
    cur_block: Block,
    /// Values that opt rules have indicated should be rematerialized
    /// in every block they are used (e.g., immediates or other
    /// "cheap-to-compute" ops).
    remat_values: &'a FxHashSet<Value>,
    /// Explicitly-unrolled value elaboration stack.
    elab_stack: Vec<ElabStackEntry>,
    /// Results from the elab stack.
    elab_result_stack: Vec<ElaboratedValue>,
    /// Explicitly-unrolled block elaboration stack.
    block_stack: Vec<BlockStackEntry>,
    /// Copies of values that have been rematerialized.
    remat_copies: FxHashMap<(Block, Value), Value>,
    /// A map from instructions to their ordering information (LUC, CP,
    /// original program order sequence and information for the LICM
    /// optimization).
    inst_ordering_info_map: &'a mut SecondaryMap<Inst, OrderingInfo>,
    /// A queue that is used to indicate the original program order
    /// of the skeleton instructions.
    skeleton_inst_order: &'a mut VecDeque<Inst>,
    /// A map from each Value to the instructions that use it.
    value_uses: SecondaryMap<Value, SmallVec<[Inst; 8]>>,
    /// A map that tracks how many dependencies (either true data dependencies
    /// or dependencies due to skeleton instructions' program order) are
    /// remainining for each instruction to get scheduled.
    ///
    /// When this count goes to 0, the instruction is inserted to the ready
    /// queue.
    dependencies_count: SecondaryMap<Inst, u8>,
    /// A priority queue that holds all ready-to-be-placed instructions.
    ///
    /// It is implemented as a Max Rank Pairing Heap, with the max element
    /// being the one that should be placed first each time.
    ready_queue: RankPairingHeap<Inst, OrderingInfo>,
    /// Stats for various events during egraph processing, to help
    /// with optimization of this infrastructure.
    stats: &'a mut Stats,
    /// Chaos-mode control-plane so we can test that we still get
    /// correct results when our heuristics make bad decisions.
    ctrl_plane: &'a mut ControlPlane,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BestEntry(Cost, Value);

impl PartialOrd for BestEntry {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for BestEntry {
    #[inline]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.cmp(&other.0).then_with(|| {
            // Note that this comparison is reversed. When costs are equal,
            // prefer the value with the bigger index. This is a heuristic that
            // prefers results of rewrites to the original value, since we
            // expect that our rewrites are generally improvements.
            self.1.cmp(&other.1).reverse()
        })
    }
}

#[derive(Clone, Copy, Debug)]
struct ElaboratedValue {
    in_block: Block,
    value: Value,
}

#[derive(Clone, Debug)]
struct LoopStackEntry {
    /// The loop identifier.
    lp: Loop,
    /// The hoist point: a block that immediately dominates this
    /// loop. May not be an immediate predecessor, but will be a valid
    /// point to place all loop-invariant ops: they must depend only
    /// on inputs that dominate the loop, so are available at (the end
    /// of) this block.
    hoist_block: Block,
    /// The depth in the scope map.
    scope_depth: u32,
}

#[derive(Clone, Debug)]
enum ElabStackEntry {
    /// Next action is to resolve this value into an elaborated inst
    /// (placed into the layout) that produces the value, and
    /// recursively elaborate the insts that produce its args.
    ///
    /// Any inserted ops should be inserted before `before`, which is
    /// the instruction demanding this value.
    Start { value: Value, before: Inst },
    /// Args have been pushed; waiting for results.
    PendingInst {
        inst: Inst,
        result_idx: usize,
        num_args: usize,
        before: Inst,
    },
}

#[derive(Clone, Debug)]
enum BlockStackEntry {
    Elaborate { block: Block, idom: Option<Block> },
    Pop,
}

impl<'a> Elaborator<'a> {
    pub(crate) fn new(
        func: &'a mut Function,
        domtree: &'a DominatorTreePreorder,
        loop_analysis: &'a LoopAnalysis,
        remat_values: &'a FxHashSet<Value>,
        inst_ordering_info_map: &'a mut SecondaryMap<Inst, OrderingInfo>,
        skeleton_inst_order: &'a mut VecDeque<Inst>,
        stats: &'a mut Stats,
        ctrl_plane: &'a mut ControlPlane,
    ) -> Self {
        let num_values = func.dfg.num_values();
        let mut value_to_best_value =
            SecondaryMap::with_default(BestEntry(Cost::infinity(), Value::reserved_value()));
        value_to_best_value.resize(num_values);
        Self {
            func,
            domtree,
            loop_analysis,
            value_to_elaborated_value: ScopedHashMap::with_capacity(num_values),
            value_to_best_value,
            loop_stack: smallvec![],
            cur_block: Block::reserved_value(),
            remat_values,
            elab_stack: vec![],
            elab_result_stack: vec![],
            block_stack: vec![],
            remat_copies: FxHashMap::default(),
            inst_ordering_info_map,
            skeleton_inst_order,
            dependencies_count: SecondaryMap::with_default(0),
            value_uses: SecondaryMap::with_default(SmallVec::new()),
            ready_queue: RankPairingHeap::single_pass_max(),
            stats,
            ctrl_plane,
        }
    }

    fn start_block(&mut self, idom: Option<Block>, block: Block) {
        trace!(
            "start_block: block {:?} with idom {:?} at loop depth {:?} scope depth {}",
            block,
            idom,
            self.loop_stack.len(),
            self.value_to_elaborated_value.depth()
        );

        // Pop any loop levels we're no longer in.
        while let Some(inner_loop) = self.loop_stack.last() {
            if self.loop_analysis.is_in_loop(block, inner_loop.lp) {
                break;
            }
            self.loop_stack.pop();
        }

        // Note that if the *entry* block is a loop header, we will
        // not make note of the loop here because it will not have an
        // immediate dominator. We must disallow this case because we
        // will skip adding the `LoopStackEntry` here but our
        // `LoopAnalysis` will otherwise still make note of this loop
        // and loop depths will not match.
        if let Some(idom) = idom {
            if let Some(lp) = self.loop_analysis.is_loop_header(block) {
                self.loop_stack.push(LoopStackEntry {
                    lp,
                    // Any code hoisted out of this loop will have code
                    // placed in `idom`, and will have def mappings
                    // inserted in to the scoped hashmap at that block's
                    // level.
                    hoist_block: idom,
                    scope_depth: (self.value_to_elaborated_value.depth() - 1) as u32,
                });
                trace!(
                    " -> loop header, pushing; depth now {}",
                    self.loop_stack.len()
                );
            }
        } else {
            debug_assert!(
                self.loop_analysis.is_loop_header(block).is_none(),
                "Entry block (domtree root) cannot be a loop header!"
            );
        }

        trace!("block {}: loop stack is {:?}", block, self.loop_stack);

        self.cur_block = block;
    }

    fn compute_best_values(&mut self) {
        let best = &mut self.value_to_best_value;

        // We can't make random decisions inside the fixpoint loop below because
        // that could cause values to change on every iteration of the loop,
        // which would make the loop never terminate. So in chaos testing
        // mode we need a form of making suboptimal decisions that is fully
        // deterministic. We choose to simply make the worst decision we know
        // how to do instead of the best.
        let use_worst = self.ctrl_plane.get_decision();

        // Do a fixpoint loop to compute the best value for each eclass.
        //
        // The maximum number of iterations is the length of the longest chain
        // of `vNN -> vMM` edges in the dataflow graph where `NN < MM`, so this
        // is *technically* quadratic, but `cranelift-frontend` won't construct
        // any such edges. NaN canonicalization will introduce some of these
        // edges, but they are chains of only two or three edges. So in
        // practice, we *never* do more than a handful of iterations here unless
        // (a) we parsed the CLIF from text and the text was funkily numbered,
        // which we don't really care about, or (b) the CLIF producer did
        // something weird, in which case it is their responsibility to stop
        // doing that.
        trace!(
            "Entering fixpoint loop to compute the {} values for each eclass",
            if use_worst {
                "worst (chaos mode)"
            } else {
                "best"
            }
        );
        let mut keep_going = true;
        while keep_going {
            keep_going = false;
            trace!(
                "fixpoint iteration {}",
                self.stats.elaborate_best_cost_fixpoint_iters
            );
            self.stats.elaborate_best_cost_fixpoint_iters += 1;

            for (value, def) in self.func.dfg.values_and_defs() {
                trace!("computing best for value {:?} def {:?}", value, def);
                let orig_best_value = best[value];

                match def {
                    ValueDef::Union(x, y) => {
                        // Pick the best of the two options based on
                        // min-cost. This works because each element of `best`
                        // is a `(cost, value)` tuple; `cost` comes first so
                        // the natural comparison works based on cost, and
                        // breaks ties based on value number.
                        best[value] = if use_worst {
                            if best[x].1.is_reserved_value() {
                                best[y]
                            } else if best[y].1.is_reserved_value() {
                                best[x]
                            } else {
                                std::cmp::max(best[x], best[y])
                            }
                        } else {
                            std::cmp::min(best[x], best[y])
                        };
                        trace!(
                            " -> best of union({:?}, {:?}) = {:?}",
                            best[x],
                            best[y],
                            best[value]
                        );
                    }
                    ValueDef::Param(_, _) => {
                        best[value] = BestEntry(Cost::zero(), value);
                    }
                    // If the Inst is inserted into the layout (which is,
                    // at this point, only the side-effecting skeleton),
                    // then it must be computed and thus we give it zero
                    // cost.
                    ValueDef::Result(inst, _) => {
                        if let Some(_) = self.func.layout.inst_block(inst) {
                            best[value] = BestEntry(Cost::zero(), value);
                        } else {
                            let inst_data = &self.func.dfg.insts[inst];
                            // N.B.: at this point we know that the opcode is
                            // pure, so `pure_op_cost`'s precondition is
                            // satisfied.
                            let cost = Cost::of_pure_op(
                                inst_data.opcode(),
                                self.func.dfg.inst_values(inst).map(|value| best[value].0),
                            );
                            best[value] = BestEntry(cost, value);
                            trace!(" -> cost of value {} = {:?}", value, cost);
                        }
                    }
                };

                // Keep on iterating the fixpoint loop while we are finding new
                // best values.
                keep_going |= orig_best_value != best[value];
            }
        }

        if cfg!(any(feature = "trace-log", debug_assertions)) {
            trace!("finished fixpoint loop to compute best value for each eclass");
            for value in self.func.dfg.values() {
                trace!("-> best for eclass {:?}: {:?}", value, best[value]);
                debug_assert_ne!(best[value].1, Value::reserved_value());
                // You might additionally be expecting an assert that the best
                // cost is not infinity, however infinite cost *can* happen in
                // practice. First, note that our cost function doesn't know
                // about any shared structure in the dataflow graph, it only
                // sums operand costs. (And trying to avoid that by deduping a
                // single operation's operands is a losing game because you can
                // always just add one indirection and go from `add(x, x)` to
                // `add(foo(x), bar(x))` to hide the shared structure.) Given
                // that blindness to sharing, we can make cost grow
                // exponentially with a linear sequence of operations:
                //
                //     v0 = iconst.i32 1    ;; cost = 1
                //     v1 = iadd v0, v0     ;; cost = 3 + 1 + 1
                //     v2 = iadd v1, v1     ;; cost = 3 + 5 + 5
                //     v3 = iadd v2, v2     ;; cost = 3 + 13 + 13
                //     v4 = iadd v3, v3     ;; cost = 3 + 29 + 29
                //     v5 = iadd v4, v4     ;; cost = 3 + 61 + 61
                //     v6 = iadd v5, v5     ;; cost = 3 + 125 + 125
                //     ;; etc...
                //
                // Such a chain can cause cost to saturate to infinity. How do
                // we choose which e-node is best when there are multiple that
                // have saturated to infinity? It doesn't matter. As long as
                // invariant (2) for optimization rules is upheld by our rule
                // set (see `cranelift/codegen/src/opts/README.md`) it is safe
                // to choose *any* e-node in the e-class. At worst we will
                // produce suboptimal code, but never an incorrectness.
            }
        }
    }

    /// Elaborate use of an eclass, inserting any needed new
    /// instructions before the given inst `before`. Should only be
    /// given values corresponding to results of instructions or
    /// blockparams.
    fn elaborate_eclass_use(&mut self, value: Value, before: Inst) -> ElaboratedValue {
        debug_assert_ne!(value, Value::reserved_value());

        // Kick off the process by requesting this result
        // value.
        self.elab_stack
            .push(ElabStackEntry::Start { value, before });

        // Now run the explicit-stack recursion until we reach
        // the root.
        self.process_elab_stack();
        debug_assert_eq!(self.elab_result_stack.len(), 1);
        self.elab_result_stack.pop().unwrap()
    }

    /// Possibly rematerialize the instruction producing the value in
    /// `arg` and rewrite `arg` to refer to it, if needed. Returns
    /// `true` if a rewrite occurred.
    fn maybe_remat_arg(
        remat_values: &FxHashSet<Value>,
        func: &mut Function,
        remat_copies: &mut FxHashMap<(Block, Value), Value>,
        insert_block: Block,
        before: Inst,
        arg: &mut ElaboratedValue,
        stats: &mut Stats,
    ) -> bool {
        // TODO (#7313): we may want to consider recursive
        // rematerialization as well. We could process the arguments of
        // the rematerialized instruction up to a certain depth. This
        // would affect, e.g., adds-with-one-constant-arg, which are
        // currently rematerialized. Right now we don't do this, to
        // avoid the need for another fixpoint loop here.
        if arg.in_block != insert_block && remat_values.contains(&arg.value) {
            let new_value = match remat_copies.entry((insert_block, arg.value)) {
                HashEntry::Occupied(o) => *o.get(),
                HashEntry::Vacant(v) => {
                    let inst = func.dfg.value_def(arg.value).inst().unwrap();
                    debug_assert_eq!(func.dfg.inst_results(inst).len(), 1);
                    let new_inst = func.dfg.clone_inst(inst);
                    func.layout.insert_inst(new_inst, before);
                    let new_result = func.dfg.inst_results(new_inst)[0];
                    *v.insert(new_result)
                }
            };
            trace!("rematerialized {} as {}", arg.value, new_value);
            arg.value = new_value;
            stats.elaborate_remat += 1;
            true
        } else {
            false
        }
    }

    fn process_elab_stack(&mut self) {
        while let Some(entry) = self.elab_stack.pop() {
            match entry {
                ElabStackEntry::Start { value, before } => {
                    debug_assert!(self.func.dfg.value_is_real(value));

                    self.stats.elaborate_visit_node += 1;

                    // Get the best option; we use `value` (latest
                    // value) here so we have a full view of the
                    // eclass.
                    trace!("looking up best value for {}", value);
                    let BestEntry(_, best_value) = self.value_to_best_value[value];
                    trace!("elaborate: value {} -> best {}", value, best_value);
                    debug_assert_ne!(best_value, Value::reserved_value());

                    if let Some(elab_val) = self.value_to_elaborated_value.get(&best_value) {
                        // Value is available; use it.
                        trace!("elaborate: value {} -> {:?}", value, elab_val);
                        self.stats.elaborate_memoize_hit += 1;
                        self.elab_result_stack.push(*elab_val);
                        continue;
                    }

                    self.stats.elaborate_memoize_miss += 1;

                    // Now resolve the value to its definition to see
                    // how we can compute it.
                    let (inst, result_idx) = match self.func.dfg.value_def(best_value) {
                        ValueDef::Result(inst, result_idx) => {
                            trace!(
                                " -> value {} is result {} of {}",
                                best_value,
                                result_idx,
                                inst
                            );
                            (inst, result_idx)
                        }
                        ValueDef::Param(in_block, _) => {
                            // We don't need to do anything to compute
                            // this value; just push its result on the
                            // result stack (blockparams are already
                            // available).
                            trace!(" -> value {} is a blockparam", best_value);
                            self.elab_result_stack.push(ElaboratedValue {
                                in_block,
                                value: best_value,
                            });
                            continue;
                        }
                        ValueDef::Union(_, _) => {
                            panic!("Should never have a Union value as the best value");
                        }
                    };

                    trace!(
                        " -> result {} of inst {:?}",
                        result_idx,
                        self.func.dfg.insts[inst]
                    );

                    // We're going to need to use this instruction
                    // result, placing the instruction into the
                    // layout. First, enqueue all args to be
                    // elaborated. Push state to receive the results
                    // and later elab this inst.
                    let num_args = self.func.dfg.inst_values(inst).count();
                    self.elab_stack.push(ElabStackEntry::PendingInst {
                        inst,
                        result_idx,
                        num_args,
                        before,
                    });

                    // Push args in reverse order so we process the
                    // first arg first.
                    // NOTE: maybe we should change the order of the arguments
                    // here, otherwise `.rev()` might not matter at all.
                    for arg in self.func.dfg.inst_values(inst).rev() {
                        debug_assert_ne!(arg, Value::reserved_value());
                        self.elab_stack
                            .push(ElabStackEntry::Start { value: arg, before });
                    }
                }

                ElabStackEntry::PendingInst {
                    inst,
                    result_idx,
                    num_args,
                    before,
                } => {
                    trace!(
                        "PendingInst: {} result {} args {} before {}",
                        inst,
                        result_idx,
                        num_args,
                        before
                    );

                    // We should have all args resolved at this
                    // point. Grab them and drain them out, removing
                    // them.
                    let arg_idx = self.elab_result_stack.len() - num_args;
                    let arg_values = &mut self.elab_result_stack[arg_idx..];

                    // Compute max loop depth.
                    //
                    // Note that if there are no arguments then this instruction
                    // is allowed to get hoisted up one loop. This is not
                    // usually used since no-argument values are things like
                    // constants which are typically rematerialized, but for the
                    // `vconst` instruction 128-bit constants aren't as easily
                    // rematerialized. They're hoisted out of inner loops but
                    // not to the function entry which may run the risk of
                    // placing too much register pressure on the entire
                    // function. This is modeled with the `.saturating_sub(1)`
                    // as the default if there's otherwise no maximum.
                    let loop_hoist_level = arg_values
                        .iter()
                        .map(|&value| {
                            // Find the outermost loop level at which
                            // the value's defining block *is not* a
                            // member. This is the loop-nest level
                            // whose hoist-block we hoist to.
                            let hoist_level = self
                                .loop_stack
                                .iter()
                                .position(|loop_entry| {
                                    !self.loop_analysis.is_in_loop(value.in_block, loop_entry.lp)
                                })
                                .unwrap_or(self.loop_stack.len());
                            trace!(
                                " -> arg: elab_value {:?} hoist level {:?}",
                                value,
                                hoist_level
                            );
                            hoist_level
                        })
                        .max()
                        .unwrap_or(self.loop_stack.len().saturating_sub(1));
                    trace!(
                        " -> loop hoist level: {:?}; cur loop depth: {:?}, loop_stack: {:?}",
                        loop_hoist_level,
                        self.loop_stack.len(),
                        self.loop_stack,
                    );

                    // We know that this is a pure inst, because
                    // non-pure roots have already been placed in the
                    // value-to-elab'd-value map, so they will not
                    // reach this stage of processing.
                    //
                    // We now must determine the location at which we
                    // place the instruction. This is the current
                    // block *unless* we hoist above a loop when all
                    // args are loop-invariant (and this op is pure).
                    let (scope_depth, before, insert_block) =
                        if loop_hoist_level == self.loop_stack.len() {
                            // Depends on some value at the current
                            // loop depth, or remat forces it here:
                            // place it at the current location.
                            self.inst_ordering_info_map[inst].before = None;
                            (
                                self.value_to_elaborated_value.depth(),
                                before,
                                self.func.layout.inst_block(before).unwrap(),
                            )
                        } else {
                            // Does not depend on any args at current
                            // loop depth: hoist out of loop.
                            self.stats.elaborate_licm_hoist += 1;
                            let data = &self.loop_stack[loop_hoist_level];
                            // `data.hoist_block` should dominate `before`'s block.
                            let before_block = self.func.layout.inst_block(before).unwrap();
                            debug_assert!(self.domtree.dominates(data.hoist_block, before_block));
                            // Determine the instruction at which we
                            // insert in `data.hoist_block`.
                            let before = self.func.layout.last_inst(data.hoist_block).unwrap();
                            self.inst_ordering_info_map[inst].before = Some(before);
                            (data.scope_depth as usize, before, data.hoist_block)
                        };

                    trace!(
                        " -> decided to place: before {} insert_block {}",
                        before,
                        insert_block
                    );

                    // Now that we have the location for the
                    // instruction, check if any of its args are remat
                    // values. If so, and if we don't have a copy of
                    // the rematerializing instruction for this block
                    // yet, create one.
                    let mut remat_arg = false;
                    for arg_value in arg_values.iter_mut() {
                        if Self::maybe_remat_arg(
                            &self.remat_values,
                            &mut self.func,
                            &mut self.remat_copies,
                            insert_block,
                            before,
                            arg_value,
                            &mut self.stats,
                        ) {
                            remat_arg = true;
                        }
                    }

                    // Now we need to place `inst` at the computed
                    // location (just before `before`). Note that
                    // `inst` may already have been placed somewhere
                    // else, because a pure node may be elaborated at
                    // more than one place. In this case, we need to
                    // duplicate the instruction (and return the
                    // `Value`s for that duplicated instance instead).
                    //
                    // Also clone if we rematerialized, because we
                    // don't want to rewrite the args in the original
                    // copy.
                    trace!("need inst {} before {}", inst, before);
                    let inst = if self.func.layout.inst_block(inst).is_some() || remat_arg {
                        // Clone the inst!
                        let new_inst = self.func.dfg.clone_inst(inst);
                        trace!(
                            " -> inst {} already has a location; cloned to {}",
                            inst,
                            new_inst
                        );
                        // Create mappings in the
                        // value-to-elab'd-value map from original
                        // results to cloned results.
                        for (&result, &new_result) in self
                            .func
                            .dfg
                            .inst_results(inst)
                            .iter()
                            .zip(self.func.dfg.inst_results(new_inst).iter())
                        {
                            let elab_value = ElaboratedValue {
                                value: new_result,
                                in_block: insert_block,
                            };
                            let best_result = self.value_to_best_value[result];
                            self.value_to_elaborated_value.insert_if_absent_with_depth(
                                best_result.1,
                                elab_value,
                                scope_depth,
                            );

                            self.value_to_best_value[new_result] = best_result;

                            trace!(
                                " -> cloned inst has new result {} for orig {}",
                                new_result,
                                result
                            );
                        }
                        new_inst
                    } else {
                        trace!(" -> no location; using original inst");
                        // Create identity mappings from result values
                        // to themselves in this scope, since we're
                        // using the original inst.
                        for &result in self.func.dfg.inst_results(inst) {
                            let elab_value = ElaboratedValue {
                                value: result,
                                in_block: insert_block,
                            };
                            let best_result = self.value_to_best_value[result];
                            self.value_to_elaborated_value.insert_if_absent_with_depth(
                                best_result.1,
                                elab_value,
                                scope_depth,
                            );
                            trace!(" -> inserting identity mapping for {}", result);
                        }
                        inst
                    };

                    //  NOTE
                    // `before` inst for the possible hoisting outside of loops.
                    assert!(
                        is_pure_for_egraph(self.func, inst),
                        "something has gone very wrong if we are elaborating effectful \
                         instructions, they should have remained in the skeleton"
                    );
                    // NOTE: Here was the insertion point of the instruction to the layout.

                    // Update the inst's arguments.
                    self.func
                        .dfg
                        .overwrite_inst_values(inst, arg_values.into_iter().map(|ev| ev.value));

                    // Now that we've consumed the arg values, pop
                    // them off the stack.
                    self.elab_result_stack.truncate(arg_idx);

                    // Push the requested result index of the
                    // instruction onto the elab-results stack.
                    self.elab_result_stack.push(ElaboratedValue {
                        in_block: insert_block,
                        value: self.func.dfg.inst_results(inst)[result_idx],
                    });
                }
            }
        }
    }

    fn elaborate_block(&mut self, elab_values: &mut Vec<Value>, idom: Option<Block>, block: Block) {
        trace!("elaborate_block: block {}", block);
        self.start_block(idom, block);

        // Iterate over the side-effecting skeleton using the linked
        // list in Layout. We will insert instructions that are
        // elaborated *before* `inst`, so we can always use its
        // next-link to continue the iteration.
        let first_inst = self.func.layout.first_inst(block);
        let mut next_inst = first_inst;
        let mut first_branch = None;

        while let Some(inst) = next_inst {
            trace!(
                "elaborating inst {} with results {:?}",
                inst,
                self.func.dfg.inst_results(inst)
            );

            // The relative order of skeleton instructions must not change,
            // since that would change program semantics. To achieve this, we
            // manually fabricate dependencies among them using the dependencies
            // count map. We basically add one dependency for each of the
            // skeleton instructions with their preceding skeleton instruction.
            self.dependencies_count[inst] += 1;

            // Record the first branch we see in the block; all
            // elaboration for args of *any* branch must be inserted
            // before the *first* branch, because the branch group
            // must remain contiguous at the end of the block.
            if self.func.dfg.insts[inst].opcode().is_branch() && first_branch == None {
                first_branch = Some(inst);
            }

            // Determine where elaboration inserts insts.
            let before = first_branch.unwrap_or(inst);
            trace!(" -> inserting before {}", before);

            elab_values.extend(self.func.dfg.inst_values(inst));
            // NOTE: the order of the arguments should not define the layout.
            for arg in elab_values.iter_mut() {
                trace!(" -> arg {}", *arg);
                // Elaborate the arg, placing any newly-inserted insts
                // before `before`. Get the updated value, which may
                // be different than the original.
                let mut new_arg = self.elaborate_eclass_use(*arg, before);
                Self::maybe_remat_arg(
                    &self.remat_values,
                    &mut self.func,
                    &mut self.remat_copies,
                    block,
                    inst,
                    &mut new_arg,
                    &mut self.stats,
                );
                trace!("   -> rewrote arg to {:?}", new_arg);
                *arg = new_arg.value;
            }

            self.func
                .dfg
                .overwrite_inst_values(inst, elab_values.drain(..));

            // We need to put the results of this instruction in the
            // map now.
            for &result in self.func.dfg.inst_results(inst) {
                trace!(" -> result {}", result);
                let best_result = self.value_to_best_value[result];
                self.value_to_elaborated_value.insert_if_absent(
                    best_result.1,
                    ElaboratedValue {
                        in_block: block,
                        value: result,
                    },
                );
            }

            next_inst = self.func.layout.next_inst(inst);
        }

        // The first skeleton instruction has no dependency due to previous
        // skeleton instructions.
        if let Some(first_inst) = first_inst {
            self.dependencies_count[first_inst] -= 1;
        }
    }

    /// Compute the critical path for all instructions in the given `block`, and
    /// construct the `value_uses` and the `dependencies_count` maps.
    ///
    /// It also initializes the ready queue with all instructions that have 0
    /// dependencies after the pass.
    fn compute_ddg_and_value_uses(&mut self, block: Block) {
        let mut next_inst = self.func.layout.last_inst(block);
        let mut inst_queue: VecDeque<Inst> = VecDeque::new();
        let mut visited_arg: SecondaryMap<Value, bool> = SecondaryMap::with_default(false);

        while let Some(inst) = next_inst {
            for arg in self.func.dfg.inst_values(inst) {
                // TODO: there probably exists a less expensive way to get unique args.
                // Skip already visited arguments in case the instruction used a
                // value multiple times.
                if visited_arg[arg] == true {
                    continue;
                }
                visited_arg[arg] = true;

                // Add the instruction to the value_uses map of its arguments.
                if !self.value_uses[arg].iter().any(|&inst| inst == inst) {
                    self.value_uses[arg].push(inst);
                }

                // Make sure that the argument comes from the result of an
                // instruction inside the current block.
                if let Some(arg_inst) = self.func.dfg.value_def(arg).inst() {
                    // Calculate the critical path for each instruction.
                    let prev_critical_path = self.inst_ordering_info_map[arg_inst].critical_path;
                    let current_path_size = self.inst_ordering_info_map[inst].critical_path + 1;
                    if current_path_size > prev_critical_path {
                        self.inst_ordering_info_map[arg_inst].critical_path = current_path_size;
                    }

                    // Construct the `dependencies_count` map.
                    self.dependencies_count[inst] += 1;

                    // Push the arguments of the instruction to the instruction
                    // queue for the breadth-first traversal of the data
                    // dependency graph.
                    inst_queue.push_back(arg_inst);
                }
            }

            next_inst = inst_queue.pop_front();
            // Initialize the ready queue with all instructions that have 0 dependencies.
            if self.dependencies_count[inst] == 0 {
                self.ready_queue
                    .push(inst, self.inst_ordering_info_map[inst]);
            }
        }
    }

    /// Put instructions back to the function's layout using the LUC and CP
    /// heuristics.
    fn schedule_insts(&mut self) {
        while let Some(inst) = self.ready_queue.pop() {
            // Update the last-use-counts of instructions.
            for arg in self.func.dfg.inst_values(inst) {
                // Remove the instruction from the argument value's users.
                self.value_uses[arg].retain(|&mut arg_user| arg_user != inst);
                // If the value has exactly one user left, increment its last-use-count,
                // and update the RankPairingHeap representing the ready queue.
                if self.value_uses[arg].len() == 1 {
                    let last_user = self.value_uses[arg].get(0).unwrap().clone();
                    self.inst_ordering_info_map[last_user].last_use_count += 1;
                    self.ready_queue
                        .update(&last_user, self.inst_ordering_info_map[last_user]);
                }
            }

            // Update the dependency counts for instructions and append ready instructions
            // to the ready queue.
            for result in self.func.dfg.inst_results(inst).iter().cloned() {
                // For each result, find all instructions that use it and decrement their
                // dependency count.
                for user in self.value_uses[result].iter().cloned() {
                    self.dependencies_count[user] -= 1;
                    // If the instruction has no dependencies left, push it in the ready queue.
                    if self.dependencies_count[user] == 0 {
                        self.ready_queue
                            .push(user, self.inst_ordering_info_map[user]);
                        // If the instruction was a skeleton instruction, decrement the dependency
                        // count of the next skeleton instruction. If the next skeleton instruction
                        // reaches zero dependencies too, continue the same operation until we find
                        // the first skeleton instruction with active dependencies, or until we
                        // elaborate the last skeleton instruction.
                        if self.skeleton_inst_order.pop_front() == Some(user) {
                            while let Some(next_skeleton_inst) = self.skeleton_inst_order.front() {
                                let next_skeleton_inst = next_skeleton_inst.clone();
                                self.dependencies_count[next_skeleton_inst] -= 1;
                                if self.dependencies_count[next_skeleton_inst] == 0 {
                                    self.skeleton_inst_order.pop_front();
                                    self.ready_queue.push(
                                        next_skeleton_inst,
                                        self.inst_ordering_info_map[next_skeleton_inst],
                                    );
                                    continue;
                                } else {
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    fn elaborate_domtree(&mut self, domtree: &DominatorTreePreorder) {
        self.block_stack.push(BlockStackEntry::Elaborate {
            block: self.func.layout.entry_block().unwrap(),
            idom: None,
        });

        // A temporary workspace for elaborate_block, allocated here to maximize the use of the
        // allocation.
        let mut elab_values = Vec::new();

        while let Some(top) = self.block_stack.pop() {
            match top {
                BlockStackEntry::Elaborate { block, idom } => {
                    self.block_stack.push(BlockStackEntry::Pop);
                    self.value_to_elaborated_value.increment_depth();

                    self.elaborate_block(&mut elab_values, idom, block);
                    self.compute_ddg_and_value_uses(block);
                    self.schedule_insts();

                    // Push children. We are doing a preorder
                    // traversal so we do this after processing this
                    // block above.
                    let block_stack_end = self.block_stack.len();
                    // NOTE: here we destroy original ordering across basic
                    // blocks — this probably doesn't mess with careful
                    // pipelining though.
                    for child in self.ctrl_plane.shuffled(domtree.children(block)) {
                        self.block_stack.push(BlockStackEntry::Elaborate {
                            block: child,
                            idom: Some(block),
                        });
                    }
                    // Reverse what we just pushed so we elaborate in
                    // original block order. (The domtree iter is a
                    // single-ended iter over a singly-linked list so
                    // we can't `.rev()` above.)
                    self.block_stack[block_stack_end..].reverse();
                }
                BlockStackEntry::Pop => {
                    self.value_to_elaborated_value.decrement_depth();
                }
            }
        }
    }

    pub(crate) fn elaborate(&mut self) {
        self.stats.elaborate_func += 1;
        self.stats.elaborate_func_pre_insts += self.func.dfg.num_insts() as u64;
        self.compute_best_values();
        self.elaborate_domtree(&self.domtree);
        self.stats.elaborate_func_post_insts += self.func.dfg.num_insts() as u64;
    }
}
