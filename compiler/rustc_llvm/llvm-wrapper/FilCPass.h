//===- FilCPass.h - Simple memory safety instrumentation -----------------===//
//
// Simplified Fil-C-inspired memory safety pass that works with standard LLVM
// IR (no flight pointers / non-integral AS0 required).
//
//===----------------------------------------------------------------------===//

#ifndef RUSTC_LLVM_FILC_PASS_H
#define RUSTC_LLVM_FILC_PASS_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class FilCPass : public PassInfoMixin<FilCPass> {
  /// When false (the default), accesses that rustc marked `!filc.safe` are left
  /// alone because Rust already proved them in bounds and pointing at live
  /// memory. When true, every access is checked regardless of what rustc said.
  bool InstrumentAll;

public:
  explicit FilCPass(bool InstrumentAll = false)
      : InstrumentAll(InstrumentAll) {}

  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM);
};

} // namespace llvm

#endif // RUSTC_LLVM_FILC_PASS_H
