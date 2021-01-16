#include <torch/csrc/jit/codegen/cuda/lower_expr_sort.h>

#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/lower_compute_at_map.h>
#include <torch/csrc/jit/codegen/cuda/lower_utils.h>
#include <torch/csrc/jit/codegen/cuda/segmenter_helper.h>

#include <unordered_map>
#include <unordered_set>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace {
// TODO: Remove
template <class T>
void print(const std::unordered_set<T>* set) {
  std::cout << "{";
  for (auto it = set->begin(); it != set->end(); it++) {
    std::cout << (*it);
    std::cout << ", ";
  }
  std::cout << "}\n";
}
} // namespace

class ExprSortPayload : public TraversalPayload {
 public:
  std::vector<IterDomain*> ca_domains;
};

class ExprSortingWithCA : public SegmentCandidateFinder {
 public:
  ExprSortingWithCA() : SegmentCandidateFinder(FusionGuard::getCurFusion()) {
    TORCH_INTERNAL_ASSERT(FusionGuard::getCurFusion() != nullptr);
  }

  ExprSortPayload* payload(SegmentedGroup* sg) {
    return sg->payload()->as<ExprSortPayload>();
  }

  bool codeGenSupportedMerge(SegmentedGroup* sg1, SegmentedGroup* sg2)
      override {
    auto domain1 = payload(sg1)->ca_domains;
    auto domain2 = payload(sg2)->ca_domains;

    if (domain1.empty() && domain2.empty()) {
      return true;
    }

    if (domain1.empty() || domain2.empty()) {
      return false;
    }

    return GpuLower::current()->caIndexMap().areMapped(domain1.back(), domain2.back());
  }

  SegmentedGroup* makeEmptyGroup() override {
    groups.push_back(
        std::make_unique<SegmentedGroup>(std::make_unique<ExprSortPayload>()));
    return groups.back().get();
  }

  SegmentedGroup* makeEmptyGroup(Expr* expr) {
    groups.push_back(
        std::make_unique<SegmentedGroup>(std::make_unique<ExprSortPayload>()));
    auto* group = groups.back().get();
    group->exprs_.push_back(expr);
    if (ir_utils::isTVOp(expr)) {
      auto out_tv = expr->outputs()[0]->as<TensorView>();
      auto* group_payload = payload(group);
      for (size_t tv_i = 0; tv_i < GpuLower::current()->caIndexMap().produce_at_map().at(out_tv);
           tv_i++) {
        group_payload->ca_domains.push_back(out_tv->axis(tv_i));
      }
    }
    return group;
  }

  SegmentedGroup* makeMergedNode(SegmentedGroup* sg1, SegmentedGroup* sg2)
      override {
    std::vector<IterDomain*> resulting_ca_axes;
    auto& domain1 = payload(sg1)->ca_domains;
    auto& domain2 = payload(sg2)->ca_domains;
    auto it1 = domain1.begin();
    auto it2 = domain2.begin();

    while (it1 != domain1.end() && it2 != domain2.end()) {
      if (it1 == domain1.end()) {
        resulting_ca_axes.push_back(*it2++);
      } else if (it2 == domain2.end()) {
        resulting_ca_axes.push_back(*it1++);
      } else if (GpuLower::current()->caIndexMap().areMapped(*it1, *it2)) {
        resulting_ca_axes.push_back(*it1);
        ++it1;
        ++it2;
      } else if (std::any_of(it1 + 1, domain1.end(), [&](IterDomain* id1) {
                   return GpuLower::current()->caIndexMap().areMapped(id1, *it2);
                 })) {
        // Increment it1, as a later iter domain matches the current one in
        // domain2
        resulting_ca_axes.push_back(*it1++);

      } else if (std::any_of(it2 + 1, domain2.end(), [&](IterDomain* id2) {
                   return GpuLower::current()->caIndexMap().areMapped(id2, *it1);
                 })) {
        // Increment it2, as a later iter domain matches the current one in
        // domain1
        resulting_ca_axes.push_back(*it2++);
      } else {
        resulting_ca_axes.push_back(*it1++);
        resulting_ca_axes.push_back(*it2++);
      }
    }

    SegmentedGroup* joined_groups =
        SegmentCandidateFinder::makeMergedNode(sg1, sg2);

    payload(joined_groups)->ca_domains = resulting_ca_axes;

    return joined_groups;
  }

  // Update in between attempts to segment. This is called once no more groups
  // can be merged together. Typically we will want to remove compute at groups
  // that have finished being grouped together. However if no gruops have been
  // merged after we've done this, we may need to stop as we could have multiple
  // disjoint groups that won't be merged.
  bool interIterUpdate() override {
    // for(auto& group : groups){
    //   std::cout<<"==============="<<std::endl;
    //   for(auto expr : group->exprs_){
    //     std::cout<<expr<<std::endl;
    //   }
    // }
    // std::cout<<"==============="<<std::endl;

    // Go through groups and lower compute at domain
    bool lowered_ca_domain = false;
    for (auto& unique_group : groups) {
      auto group = unique_group.get();
      IterDomain* g_last_id = nullptr;
      if (payload(group)->ca_domains.size() > 0) {
        g_last_id = payload(group)->ca_domains.back();
      }
      if (g_last_id == nullptr) {
        continue;
      }

      bool matching_neighbor = false;
      for (auto neighbor : group->getNeighbors()) {
        if (matching_neighbor) {
          break;
        }
        for (auto p_id : payload(neighbor)->ca_domains) {
          if (GpuLower::current()->caIndexMap().areMapped(p_id, g_last_id)) {
            matching_neighbor = true;
            break;
          }
        }
      }

      if (!matching_neighbor) {
        // std::cout<<"Lowering group: ";
        // for(auto expr : unique_group->exprs_){
        //   std::cout<<"T"<<expr->outputs()[0]->name()<<", ";
        // }std::cout<<std::endl;
        payload(group)->ca_domains.pop_back();
        lowered_ca_domain = true;
      }
    }

    // If we couldn't lower compute at domain any further, and we haven't merged
    // any new groups since the last time we were called, make sure we're done.
    if (!lowered_ca_domain && n_groups == groups.size()) {
      // Make sure none of the groups are still connected, as that would mean we
      // should have been able to merge them.

      TORCH_INTERNAL_ASSERT(
          std::all_of(
              groups.begin(),
              groups.end(),
              [](std::unique_ptr<SegmentedGroup>& sg) {
                return sg->producer_edges.empty() && sg->consumer_edges.empty();
              }),
          "Couldn't succcessfully sort out the fusion expressions. ",
          "There are remaining connections of the heirarchical segmentation which should have been ",
          "flattened to a single ordered group, or disjoint ordered groups.");

      // Successfully finished
      return false;
    }

    // Initialize n_groups if this is the first pass.
    if (n_groups == 0 && groups.size() > 0) {
      n_groups = groups.size();
    }

    n_groups = groups.size();
    // Not done, continue.
    return true;
  }

  // Track how many groups we have from iteration to iteration so we can track
  // when we've stopped merging nodes.
  size_t n_groups = 0;

};

std::vector<Expr*> reorderExprsTest() {
  ExprSortingWithCA sorter;
  sorter.segment();
  auto groups = sorter.getGroups();
  TORCH_INTERNAL_ASSERT(
      groups.size() > 0,
      "Error during expression sorting, no expressions produced.");

  // We could have multiple groups if they're disjoint. Simply flatten them in
  // order as they could be in any order.
  std::vector<Expr*> exprs;
  for (auto group : groups) {
    exprs.insert(exprs.end(), group->exprs_.begin(), group->exprs_.end());
  }
  return exprs;
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
