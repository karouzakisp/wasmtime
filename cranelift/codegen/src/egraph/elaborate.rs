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
    _loop_analysis: &'a LoopAnalysis,
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
    _loop_stack: SmallVec<[LoopStackEntry; 8]>,
    /// The current block into which we are elaborating.
    _cur_block: Block,
    /// Values that opt rules have indicated should be rematerialized
    /// in every block they are used (e.g., immediates or other
    /// "cheap-to-compute" ops).
    _remat_values: &'a FxHashSet<Value>,
    /// Results from the elab stack.
    _elab_result_stack: Vec<ElaboratedValue>,
    /// Explicitly-unrolled block elaboration stack.
    block_stack: Vec<BlockStackEntry>,
    /// Copies of values that have been rematerialized.
    _remat_copies: FxHashMap<(Block, Value), Value>,
    /// A map from instructions to their ordering information (LUC, CP,
    /// original program order sequence and information for the LICM
    /// optimization).
    inst_ordering_info_map: &'a mut SecondaryMap<Inst, OrderingInfo>,
    /// A queue that is used to indicate the original program order
    /// of the skeleton instructions.
    skeleton_inst_order: &'a mut VecDeque<Inst>,
    /// A map from each Value to the instructions that use it.
    value_users: SecondaryMap<Value, SmallVec<[Inst; 8]>>,
    /// A map that tracks how many dependencies (either true data dependencies
    /// or dependencies due to skeleton instructions' program order) are
    /// remainining for each instruction to get scheduled.
    ///
    /// When this count goes to 0, the instruction is inserted to the ready
    /// queue.
    dependencies_count: SecondaryMap<Inst, u16>,
    /// A priority queue that holds all ready-to-be-placed instructions.
    ///
    /// It is implemented as a Max Rank Pairing Heap, with the max element
    /// being the one that should be placed first each time.
    ready_queue: RankPairingHeap<Inst, OrderingInfo>,
    /// Stats for various events during egraph processing, to help
    /// with optimization of this infrastructure.
    //
    // TODO: Check if the `elaborate_visit_node` field still makes sense. In
    // general, check if the `Stats` fields need any changes after the
    // heuristics scheduling implementation.
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
            _loop_analysis: loop_analysis,
            value_to_elaborated_value: ScopedHashMap::with_capacity(num_values),
            value_to_best_value,
            _loop_stack: smallvec![],
            _cur_block: Block::reserved_value(),
            _remat_values: remat_values,
            _elab_result_stack: vec![],
            block_stack: vec![],
            _remat_copies: FxHashMap::default(),
            inst_ordering_info_map,
            skeleton_inst_order,
            dependencies_count: SecondaryMap::with_default(0),
            value_users: SecondaryMap::with_default(SmallVec::new()),
            ready_queue: RankPairingHeap::single_pass_max(),
            stats,
            ctrl_plane,
        }
    }

    // NOTE: this is unecessary until we re-enable the LICM optimization
    fn _start_block(&mut self, idom: Option<Block>, block: Block) {
        trace!(
            "start_block: block {:?} with idom {:?} at loop depth {:?} scope depth {}",
            block,
            idom,
            self._loop_stack.len(),
            self.value_to_elaborated_value.depth()
        );

        // Pop any loop levels we're no longer in.
        while let Some(inner_loop) = self._loop_stack.last() {
            if self._loop_analysis.is_in_loop(block, inner_loop.lp) {
                break;
            }
            self._loop_stack.pop();
        }

        // Note that if the *entry* block is a loop header, we will
        // not make note of the loop here because it will not have an
        // immediate dominator. We must disallow this case because we
        // will skip adding the `LoopStackEntry` here but our
        // `LoopAnalysis` will otherwise still make note of this loop
        // and loop depths will not match.
        if let Some(idom) = idom {
            if let Some(lp) = self._loop_analysis.is_loop_header(block) {
                self._loop_stack.push(LoopStackEntry {
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
                    self._loop_stack.len()
                );
            }
        } else {
            debug_assert!(
                self._loop_analysis.is_loop_header(block).is_none(),
                "Entry block (domtree root) cannot be a loop header!"
            );
        }

        trace!("block {}: loop stack is {:?}", block, self._loop_stack);

        self._cur_block = block;
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

    // TODO: re-enable when everything else is functioning
    /// Possibly rematerialize the instruction producing the value in
    /// `arg` and rewrite `arg` to refer to it, if needed. Returns
    /// `true` if a rewrite occurred.
    fn _maybe_remat_arg(
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

    // Remove the skeleton instructions left inside the block, while also
    // manufacturing the dependencies among them due to their original program
    // order.
    fn add_skeleton_dependencies(&mut self, block: Block) {
        let first_inst = self.func.layout.first_inst(block);
        let mut next_inst = first_inst;

        while let Some(inst) = next_inst {
            next_inst = self.func.layout.next_inst(inst);

            // Create the dependencies among skeleton instructions.
            self.dependencies_count[inst] += 1;

            // Remove all the skeleton instructions except the last one.
            if next_inst.is_some() {
                self.func.layout.remove_inst(inst);
            }
        }

        // FIXME: only needed for debugging... ////////////////////////////////
        if let Some(inst) = next_inst {
            assert!(self.func.dfg.insts[inst].opcode().is_terminator());
        }
        ///////////////////////////////////////////////////////////////////////

        // The first skeleton instruction has no dependency since there are no
        // previous skeleton instructions in the block.
        if let Some(first_inst) = first_inst {
            if !self.func.dfg.insts[first_inst].opcode().is_terminator() {
                self.dependencies_count[first_inst] -= 1;
            }
        }
    }

    /// Compute the critical path for all instructions in the given `block`, and
    /// construct the `value_uses` and the `dependencies_count` maps.
    ///
    /// It also initializes the ready queue with all instructions that have 0
    /// dependencies after the pass.
    fn compute_ddg_and_value_uses(&mut self, block: Block) {
        assert_eq!(self.func.layout.block_insts(block).count(), 1);
        let mut next_inst = self.func.layout.last_inst(block);
        let block_terminator = next_inst.unwrap();
        let mut inst_queue: VecDeque<Inst> = VecDeque::new();

        let mut skeleton_already_inserted = false;

        while let Some(inst) = next_inst {
            // A map to filter out duplicate visits in case an instruction uses
            // the same value in multiple arguments.
            let mut inst_arg_already_visited: SecondaryMap<Value, bool> =
                SecondaryMap::with_default(false);
            for arg in self.func.dfg.inst_values(inst) {
                // TODO: there probably exists a less expensive way to get unique args.
                // Skip already visited arguments in case the instruction uses
                // the same argument value multiple times. Basically, if we
                // have `add x0, x1, x1` we don't want to add two dependencies
                // in the instruction because of `x1`.
                if inst_arg_already_visited[arg] {
                    continue;
                }
                inst_arg_already_visited[arg] = true;

                // Add the instruction to the value_uses map of its arguments.
                if !self.value_users[arg]
                    .iter()
                    .any(|&user_inst| user_inst == inst)
                {
                    self.value_users[arg].push(inst);
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
            // FIXME: check special cases with skeleton instructions here
            if self.dependencies_count[inst] == 0 && inst != block_terminator {
                if is_pure_for_egraph(self.func, inst) {
                    self.ready_queue
                        .push(inst, self.inst_ordering_info_map[inst]);
                } else if Some(&inst) == self.skeleton_inst_order.front()
                    && !skeleton_already_inserted
                    && inst != block_terminator
                {
                    self.ready_queue
                        .push(inst, self.inst_ordering_info_map[inst]);
                    skeleton_already_inserted = true;
                }
            }
        }
        // FIXME: is this check correct?
        // assert!(
        //     !self.ready_queue.is_empty(),
        //     "computeDDG: found empty ready_queue"
        // );
    }

    /// Put instructions back to the function's layout using the LUC and CP
    /// heuristics.
    fn schedule_insts(&mut self, block: Block) {
        // Take the block terminator. It should be the only instruction left
        // inside the block, and be a terminator.
        let block_terminator = self.func.layout.first_inst(block).unwrap();
        assert_eq!(self.func.layout.block_insts(block).count(), 1);
        assert!(self.func.dfg.insts[block_terminator]
            .opcode()
            .is_terminator());
        // FIXME: only needed for debugging... ////////////////////////////////
        let mut elaborated_instructions: SecondaryMap<Inst, bool> =
            SecondaryMap::with_default(false);
        let mut instruction_in_ready_queue: SecondaryMap<Inst, bool> =
            SecondaryMap::with_default(false);
        // FIXME: is this check correct?
        // assert!(
        //     !self.ready_queue.is_empty(),
        //     "schedule_insts: found empty ready_queue"
        // );
        ///////////////////////////////////////////////////////////////////////

        // NOTE: The ready queue can only be empty here if (and only if) the
        // block ought to have just the block terminator inside it. Maybe
        // formulate an analogous assertion.

        while let Some(mut inst_to_insert) = self.ready_queue.pop() {
            // FIXME: only needed for debugging... ////////////////////////////
            assert!(inst_to_insert != block_terminator);
            ///////////////////////////////////////////////////////////////////

            // If all results of the to-be-inserted instruction have already
            // been created through instruction elaborations, we can reuse them,
            // so inserting the instruction would have no effect for us, given
            // that the instruction was not in the side-effecting skeleton.
            //
            // We hence can skip those instructions' insertion, but we still
            // have to decrement dependency counts for its results' users. We
            // also have to remove each result value from the value_uses maps of
            // its results' users, and possibly change LUC fields accordingly.
            let redundant_inst = self
                .func
                .dfg
                .inst_results(inst_to_insert)
                .iter()
                .all(|inst_result| self.value_to_elaborated_value.get(inst_result).is_some())
                && is_pure_for_egraph(self.func, inst_to_insert);

            // TODO: In case it the instruction got optimized through LICM (got
            // hoisted out of a loop), change its layout placement accordingly.
            //
            // NOTE: Remove the `before` field from `inst_ordering_info_map`
            // since we are now going to compute it eagerly inside here.
            let (scope_depth, before, insert_block) = (
                self.value_to_elaborated_value.depth(),
                block_terminator,
                self.func.layout.inst_block(block_terminator).unwrap(),
            );

            // TODO: Now that we have the location for the instruction, check
            // if any of its args are remat values. If so, and if we don't have
            // a copy of the rematerializing instruction for this block yet,
            // create one.
            // TODO: =========== rematerialization ===================
            // Check if the pure instructions have only 1 result or more.
            // After remat update all the user args of that arg that just
            // rematerialized.
            // And update the value_uses map. Inserting a Map Entry for the
            // generated value along with the users of that value.
            // For the newly generated instruction we don't decremenent dependency
            // count of other instructions that are the results of the users of the results
            let remat_arg = false;

            // Now we need to place `inst` at the computed location (just
            // before `before`). Note that `inst` may already have been
            // placed somewhere else, because a pure node may be elaborated
            // at more than one place. In this case, we need to duplicate the
            // instruction.
            //
            // NOTE: We must see if this duplication will necesitate changes in
            // the `value_uses` map, the ordering information, the
            // `dependencies_count` map etc. It is possible that this
            // duplication might move in the `compute_ddg_and_value_uses` pass!
            trace!("need inst {} before {}", inst_to_insert, before);
            if !redundant_inst {
                inst_to_insert =
                    if self.func.layout.inst_block(inst_to_insert).is_some() || remat_arg {
                        // Clone the inst!
                        let new_inst = self.func.dfg.clone_inst(inst_to_insert);

                        // FIXME: CHECK again self.dependencies_count[new_inst] = self.dependencies_count[inst_to_insert];

                        trace!(
                            " -> inst {} already has a location; cloned to {}",
                            inst_to_insert,
                            new_inst
                        );

                        // Create mappings in the value-to-elab'd-value map from
                        // original results to cloned results, and generate the
                        // necessary value_users maps.
                        let result_pairs: Vec<(Value, Value)> = self
                            .func
                            .dfg
                            .inst_results(inst_to_insert)
                            .iter()
                            .cloned()
                            .zip(self.func.dfg.inst_results(new_inst).iter().cloned())
                            .collect();
                        for (result, new_result) in result_pairs.iter() {
                            // Clone the value_users for each newly-generated result using the maps
                            // from the old results.
                            self.value_users[*new_result] = self.value_users[*result].clone();

                            // Overwrite the arguments of the users of each old result with
                            // the respective new result.
                            for user_inst in self.value_users[*result].iter().cloned() {
                                let user_args: Vec<_> =
                                    self.func.dfg.inst_values(user_inst).into_iter().collect();
                                self.func.dfg.overwrite_inst_values(
                                    user_inst,
                                    user_args.into_iter().map(|user_arg| {
                                        if user_arg == *result {
                                            *new_result
                                        } else {
                                            user_arg
                                        }
                                    }),
                                );
                            }

                            let elab_value = ElaboratedValue {
                                value: *new_result,
                                in_block: insert_block,
                            };
                            let best_result = self.value_to_best_value[*result];
                            self.value_to_elaborated_value.insert_if_absent_with_depth(
                                best_result.1,
                                elab_value,
                                scope_depth,
                            );

                            // NOTE: Understand why this is correct...
                            // Shouldn't `best_result` be `elab_value`?
                            self.value_to_best_value[*new_result] = best_result;

                            trace!(
                                " -> cloned inst has new result {} for orig {}",
                                new_result,
                                result
                            );
                        }
                        new_inst
                    } else {
                        trace!(" -> no location; using original inst");
                        // Create identity mappings from result values to themselves
                        // in this scope, since we're using the original inst.
                        for &result in self.func.dfg.inst_results(inst_to_insert) {
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
                        inst_to_insert
                    };

                let elaborated_args: Vec<Value> = self
                    .func
                    .dfg
                    .inst_values(inst_to_insert)
                    .map(|arg| {
                        let best_arg = self.value_to_best_value[arg].1;
                        match self.func.dfg.value_def(best_arg) {
                            ValueDef::Union(..) => {
                                panic!("egraph union node found at line 725!");
                            }
                            _ => {}
                        };
                        if self.func.dfg.value_def(arg).inst().is_some() {
                            let elab_value =
                                self.value_to_elaborated_value.get(&best_arg).unwrap().value;
                            match self.func.dfg.value_def(elab_value) {
                                ValueDef::Union(..) => {
                                    panic!("egraph union node found at line 734!");
                                }
                                _ => {}
                            };
                            elab_value
                        } else {
                            best_arg
                        }
                    })
                    .collect();

                // FIXME: only for debugging //////////////////////////////////
                self.func
                    .dfg
                    .inst_results(inst_to_insert)
                    .iter()
                    .for_each(|&result| match self.func.dfg.value_def(result) {
                        ValueDef::Union(..) => {
                            panic!("egraph union node found at line 751!");
                        }
                        _ => {}
                    });

                assert!(
                    elaborated_instructions[inst_to_insert] == false,
                    "We already inserted this instruction in this block through the ready queue!",
                );
                ///////////////////////////////////////////////////////////////

                self.func
                    .dfg
                    .overwrite_inst_values(inst_to_insert, elaborated_args.into_iter());

                // Insert the instruction to the layout.
                self.func
                    .layout
                    .insert_inst(inst_to_insert, block_terminator);
            }

            // NOTE: The instruction is either inserted to the layout or it was redundant_inst,
            // because all the results were already present.
            let inserted_inst = inst_to_insert;

            // FIXME: only needed for debugging... ////////////////////////////
            elaborated_instructions[inserted_inst] = true;
            ///////////////////////////////////////////////////////////////////

            // Update the LUC (last-use-counts) of instructions.
            for arg in self.func.dfg.inst_values(inserted_inst) {
                // Remove the instruction from the argument value's users.
                self.value_users[arg].retain(|&mut arg_user| arg_user != inserted_inst);
                // If the value has exactly one user left, increment its last-use-count,
                // and update the RankPairingHeap representing the ready queue.
                if self.value_users[arg].len() == 1 {
                    let last_user = self.value_users[arg].get(0).unwrap().clone();
                    self.inst_ordering_info_map[last_user].last_use_count += 1;
                    self.ready_queue
                        .update(&last_user, self.inst_ordering_info_map[last_user]);
                }
            }

            let mut skeleton_already_inserted = false;

            // If the instruction was a skeleton instruction, pop it from the
            // `skeleton_inst_order` queue and decrement the dependency count
            // of the next skeleton instruction (if it exists). If the next
            // skeleton instruction reaches zero dependencies (and is not the
            // block terminator), put it in the ready queue too, indicating that
            // a skeleton instruction has just been inserted in it, using
            // `skeleton_already_pushed`.
            if self.skeleton_inst_order.front() == Some(&inserted_inst) {
                self.skeleton_inst_order.pop_front();
                if let Some(next_skeleton_inst) = self.skeleton_inst_order.front() {
                    let next_skeleton_inst = next_skeleton_inst.clone();
                    self.dependencies_count[next_skeleton_inst] -= 1;
                    if self.dependencies_count[next_skeleton_inst] == 0
                        && next_skeleton_inst != block_terminator
                    {
                        self.ready_queue.push(
                            next_skeleton_inst,
                            self.inst_ordering_info_map[next_skeleton_inst],
                        );
                        skeleton_already_inserted = true;
                        // FIXME: only needed for debugging... ////////////////
                        assert!(!instruction_in_ready_queue[next_skeleton_inst]);
                        instruction_in_ready_queue[next_skeleton_inst] = true;
                        ///////////////////////////////////////////////////////
                    }
                }
            }

            // Update the dependency counts for all the users of the results
            // generated by the instruction that was just inserted into the
            // layout. If any instruction ends up with zero dependencies, try to
            // insert it to the ready queue.
            for result in self.func.dfg.inst_results(inserted_inst).iter().cloned() {
                // For each result, find all instructions that use it and
                // decrement their dependency count.
                for user_inst in self.value_users[result].iter().cloned() {
                    if self.dependencies_count[user_inst] != 0 {
                        self.dependencies_count[user_inst] -= 1;
                    }
                    // If the instruction has no dependencies left and is not
                    // the block terminator, try to insert it to the ready
                    // queue. The insertion will succeed only if the instruction
                    // is pure, or in case it's in the block's skeleton, if it
                    // is the next skeleton in order and hasn't already been
                    // inserted.
                    if self.dependencies_count[user_inst] == 0 && user_inst != block_terminator {
                        if is_pure_for_egraph(self.func, user_inst) {
                            self.ready_queue
                                .push(user_inst, self.inst_ordering_info_map[user_inst]);
                            // FIXME: only needed for debugging... ////////////
                            assert!(
                                user_inst != inserted_inst,
                                "We already have inserted inst to the layout"
                            );
                            assert!(!instruction_in_ready_queue[user_inst]);
                            instruction_in_ready_queue[user_inst] = true;
                            ///////////////////////////////////////////////////
                        } else if !skeleton_already_inserted
                            && Some(&user_inst) == self.skeleton_inst_order.front()
                        {
                            self.ready_queue
                                .push(user_inst, self.inst_ordering_info_map[user_inst]);
                            // FIXME: only needed for debugging... ////////////
                            assert!(!instruction_in_ready_queue[user_inst]);
                            instruction_in_ready_queue[user_inst] = true;
                            ///////////////////////////////////////////////////
                        }
                    }
                }
            }
        }

        // FIXME: Update the block terminator's arguments with elaborated values.
        let terminator_elab_args: Vec<Value> = self
            .func
            .dfg
            .inst_values(block_terminator)
            .map(|arg| {
                let best_value = self.value_to_best_value[arg].1;
                self.value_to_elaborated_value
                    .get(&best_value)
                    .unwrap()
                    .value
            })
            .collect();
        self.func
            .dfg
            .overwrite_inst_values(block_terminator, terminator_elab_args.into_iter());

        // FIXME: only needed for debugging... ////////////////////////////////
        self.func
            .dfg
            .inst_args(block_terminator)
            .iter()
            .for_each(|&arg| match self.func.dfg.value_def(arg) {
                ValueDef::Union(..) => panic!("Terminator has union param"),
                _ => {}
            });
        ///////////////////////////////////////////////////////////////////////
    }

    fn elaborate_domtree(&mut self, domtree: &DominatorTreePreorder) {
        self.block_stack.push(BlockStackEntry::Elaborate {
            block: self.func.layout.entry_block().unwrap(),
            idom: None,
        });

        while let Some(top) = self.block_stack.pop() {
            match top {
                BlockStackEntry::Elaborate { block, idom } => {
                    self.block_stack.push(BlockStackEntry::Pop);
                    self.value_to_elaborated_value.increment_depth();

                    self.add_skeleton_dependencies(block);
                    self.compute_ddg_and_value_uses(block);
                    // TODO: We might have to reset ordering info for every block.
                    self.schedule_insts(block);

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
