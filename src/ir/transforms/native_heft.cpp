#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <iostream>

#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/function.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/expr.h"

namespace pypto {
namespace ir {
namespace pass {

struct TaskBlock {
    size_t start_idx;
    size_t end_idx;
    std::string kernel_name;
    double weight;
    std::unordered_set<size_t> predecessors;
    std::unordered_set<size_t> successors;
    double rank_u = -1.0;
};

class HeftSchedulerMutator : public IRMutator {
public:
    StmtPtr VisitStmt_(const SeqStmtsPtr& op) override {
        auto mutated_stmt = IRMutator::VisitStmt_(op);
        auto seq = As<SeqStmts>(mutated_stmt);

        if (!seq || seq->stmts_.size() <= 1) return mutated_stmt;

        const auto& stmts = seq->stmts_;
        size_t n = stmts.size();

        std::vector<TaskBlock> tasks;
        std::unordered_map<const Var*, size_t> var_to_task_idx;

        for (size_t i = 0; i < n; ) {
            auto assign = As<AssignStmt>(stmts[i]);
            if (!assign) {
                TaskBlock tb{i, i, "non_assign", 0.0, {}, {}, -1.0};
                tasks.push_back(tb);
                i++;
                continue;
            }

            if (auto submit = As<Submit>(assign->value_)) {
                TaskBlock tb;
                tb.start_idx = i;
                tb.kernel_name = submit->op_->name_;
                //std::cout << tb.kernel_name << std::endl;
                
                if (tb.kernel_name.find("merge") != std::string::npos) tb.weight = 15.0;
                else if (tb.kernel_name.find("branch") != std::string::npos || 
                         tb.kernel_name.find("bg") != std::string::npos) tb.weight = 10.0;
                else tb.weight = 5.0;

                const Var* submit_tmp_var = nullptr;
                if (auto var = AsVarLike(assign->var_)) {
                    submit_tmp_var = var.get();
                    var_to_task_idx[submit_tmp_var] = tasks.size();
                }

                size_t j = i + 1;
                while (j < n) {
                    if (auto next_assign = As<AssignStmt>(stmts[j])) {
                        if (auto get_item = As<TupleGetItemExpr>(next_assign->value_)) {
                            if (auto tuple_var = AsVarLike(get_item->tuple_)) {
                                if (tuple_var.get() == submit_tmp_var) {
                                    if (auto out_var = AsVarLike(next_assign->var_)) {
                                        var_to_task_idx[out_var.get()] = tasks.size();
                                    }
                                    j++;
                                    continue;
                                }
                            }
                        }
                    }
                    break; 
                }
                
                tb.end_idx = j - 1;
                
                for (const auto& dep_expr : submit->deps_) {
                    if (auto dep_var = AsVarLike(dep_expr)) {
                        auto it = var_to_task_idx.find(dep_var.get());
                        if (it != var_to_task_idx.end()) {
                            tb.predecessors.insert(it->second);
                        }
                    }
                }
                
                tasks.push_back(tb);
                i = j;
            } else {
                TaskBlock tb{i, i, "assign", 0.0, {}, {}, -1.0};
                if (auto var = AsVarLike(assign->var_)) {
                    var_to_task_idx[var.get()] = tasks.size();
                }
                tasks.push_back(tb);
                i++;
            }
        }

        size_t num_tasks = tasks.size();
        for (size_t t = 0; t < num_tasks; ++t) {
            for (size_t pred : tasks[t].predecessors) {
                tasks[pred].successors.insert(t);
            }
        }

        std::cout << "[DAG Analysis]" << std::endl;
        for (size_t t = 0; t < num_tasks; ++t) {
            if (tasks[t].kernel_name != "assign" && tasks[t].kernel_name != "non_assign") {
                std::cout << "  Task: " << tasks[t].kernel_name << " | Depends on: ";
                if (tasks[t].predecessors.empty()) std::cout << "<none>";
                for (size_t p : tasks[t].predecessors) std::cout << tasks[p].kernel_name << " ";
                std::cout << std::endl;
            }
        }

        std::function<double(size_t)> compute_rank = [&](size_t u) {
            if (tasks[u].rank_u >= 0) return tasks[u].rank_u;
            double max_succ_rank = 0.0;
            for (size_t succ : tasks[u].successors) {
                max_succ_rank = std::max(max_succ_rank, compute_rank(succ));
            }
            tasks[u].rank_u = tasks[u].weight + max_succ_rank;
            return tasks[u].rank_u;
        };

        for (size_t i = 0; i < num_tasks; ++i) compute_rank(i);

        std::vector<size_t> indices(num_tasks);
        for (size_t i = 0; i < num_tasks; ++i) indices[i] = i;

        std::stable_sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
            return tasks[a].rank_u > tasks[b].rank_u; 
        });

        std::vector<size_t> new_order;
        new_order.reserve(num_tasks);
        std::unordered_set<size_t> scheduled;

        while (new_order.size() < num_tasks) {
            bool progressed = false;
            for (size_t idx : indices) {
                if (scheduled.count(idx)) continue;

                bool ready = true;
                for (size_t pred : tasks[idx].predecessors) {
                    if (!scheduled.count(pred)) {
                        ready = false;
                        break;
                    }
                }

                if (ready) {
                    new_order.push_back(idx);
                    scheduled.insert(idx);
                    progressed = true;
                    break; 
                }
            }
            
            if (!progressed) {
                for (size_t i = 0; i < num_tasks; ++i) {
                    if (!scheduled.count(i)) {
                        new_order.push_back(i);
                        scheduled.insert(i);
                    }
                }
                break;
            }
        }

        std::vector<StmtPtr> new_stmts;
        std::cout << "\n[Final Schedule Order]" << std::endl;
        for (size_t task_idx : new_order) {
            const auto& tb = tasks[task_idx];
            if (tb.kernel_name != "assign" && tb.kernel_name != "non_assign") {
                 std::cout << "  -> " << tb.kernel_name << " (rank_u: " << tb.rank_u << ")" << std::endl;
            }
            for (size_t stmt_idx = tb.start_idx; stmt_idx <= tb.end_idx; ++stmt_idx) {
                new_stmts.push_back(stmts[stmt_idx]);
                std::cout << stmt_idx << " " << stmts[stmt_idx] << std::endl;
            }
        }

        return std::make_shared<SeqStmts>(new_stmts, seq->span_);
    }
};

Pass NativeHeftScheduler() {
    return CreateFunctionPass(
        [](const FunctionPtr& func) -> FunctionPtr {
            if (func->func_type_ != FunctionType::Orchestration) {
                return func;
            }
            HeftSchedulerMutator mutator;
            return mutator.VisitFunction(func);
        },
        "NativeHeftScheduler",
        PassProperties{}
    );
}

} // namespace pass
} // namespace ir
} // namespace pypto