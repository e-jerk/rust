//===- FilCPass.cpp - Simple memory safety instrumentation ---------------===//
//
// Simplified Fil-C-inspired memory safety pass for Rust that works with
// standard LLVM IR. Unlike fil-c's Pizlonator, this pass does not require
// flight pointers in the IR - it instruments standard pointer operations with
// runtime checks.
//
// Approach:
// - Runtime intercepts malloc/free/mmap/munmap via libc symbol interposition
// - Pass inserts bounds / use-after-free checks before loads/stores
// - Stack allocas register type metadata when useful
//
// Ported from the Zig fil-c-integration simplified pass (ZigFilCPass).
//
// Hybrid mode
// -----------
// Rust already proves most memory accesses safe at compile time, so checking
// them again at runtime is pure overhead. rustc therefore tags every access it
// knows the borrow checker covers with `!filc.safe`, and by default this pass
// only instruments the rest: raw pointer dereferences, atomics, and anything
// that reached the module without a tag (inline asm output, C code compiled
// into the same LTO unit, IR synthesized by earlier passes, ...).
//
// The tagging is deliberately opt-out rather than opt-in. An access that loses
// its `!filc.safe` tag - because an optimization merged it with an untagged one,
// say - gets checked, which costs performance but never safety. Pass
// `-Zfilc-instrument-all` to ignore the tags and check everything, which is
// useful for measuring the speedup or for chasing a suspected gap in coverage.
//
//===----------------------------------------------------------------------===//

#include "FilCPass.h"

#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/InstrTypes.h>
#include <llvm/IR/Instruction.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/raw_ostream.h>

#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace llvm;

static cl::opt<bool> Verbose("filc-verbose", cl::desc("Make FilC verbose"),
                             cl::Hidden, cl::init(false));

namespace {

/// Type id handed to the runtime when an access has no meaningful type, such as
/// the byte copy behind `memcpy`. The runtime treats it as "any type".
constexpr uint32_t UntypedAccess = 0;

/// Returns true if the address of `AI` can be reached by anything other than a
/// direct access to the alloca itself. Only escaping stack slots need to be
/// registered with the runtime: an unescaped one can never be the target of a
/// raw pointer, so no check will ever ask about it.
bool allocaMayEscape(AllocaInst *AI) {
  SmallVector<const Value *, 8> Worklist;
  SmallPtrSet<const Value *, 8> Visited;

  Worklist.push_back(AI);
  Visited.insert(AI);

  while (!Worklist.empty()) {
    const Value *V = Worklist.pop_back_val();

    for (const Use &U : V->uses()) {
      const User *Usr = U.getUser();

      if (isa<LoadInst>(Usr))
        continue;

      if (auto *SI = dyn_cast<StoreInst>(Usr)) {
        // Storing *through* the pointer keeps the address local; storing the
        // address itself hands it to whoever reads that memory later.
        if (SI->getValueOperand() == V)
          return true;
        continue;
      }

      if (isa<GetElementPtrInst>(Usr) || isa<BitCastInst>(Usr) ||
          isa<AddrSpaceCastInst>(Usr) || isa<PHINode>(Usr) ||
          isa<SelectInst>(Usr)) {
        if (Visited.insert(Usr).second)
          Worklist.push_back(Usr);
        continue;
      }

      if (auto *II = dyn_cast<IntrinsicInst>(Usr)) {
        switch (II->getIntrinsicID()) {
        case Intrinsic::lifetime_start:
        case Intrinsic::lifetime_end:
        case Intrinsic::invariant_start:
        case Intrinsic::invariant_end:
        case Intrinsic::dbg_declare:
        case Intrinsic::dbg_value:
        case Intrinsic::dbg_label:
          continue;
        default:
          return true;
        }
      }

      return true;
    }
  }

  return false;
}

class Instrumenter {
  Module &M;
  LLVMContext &C;
  DataLayout DL;
  bool InstrumentAll;

  Type *VoidTy;
  IntegerType *Int8Ty;
  IntegerType *Int32Ty;
  IntegerType *Int64Ty;
  IntegerType *IntPtrTy;
  PointerType *PtrTy;

  unsigned FilCSafeMD;

  FunctionCallee CheckRead;
  FunctionCallee CheckWrite;
  FunctionCallee SetTypeId;
  FunctionCallee GetTypeId;

  std::unordered_map<Type *, uint32_t> TypeIds;
  uint32_t NextTypeId = 1;
  std::unordered_set<AllocaInst *> TrackedAllocas;

  unsigned NumChecked = 0;
  unsigned NumSkipped = 0;

public:
  Instrumenter(Module &M, bool InstrumentAll)
      : M(M), C(M.getContext()), DL(M.getDataLayout()),
        InstrumentAll(InstrumentAll) {
    VoidTy = Type::getVoidTy(C);
    Int8Ty = Type::getInt8Ty(C);
    Int32Ty = Type::getInt32Ty(C);
    Int64Ty = Type::getInt64Ty(C);
    PtrTy = PointerType::get(C, 0);

    unsigned PtrBits = DL.getPointerSizeInBits(0);
    IntPtrTy = Type::getIntNTy(C, PtrBits);

    FilCSafeMD = C.getMDKindID("filc.safe");
  }

  void run() {
    if (Verbose) {
      errs() << "[FilC] Instrumenting module: " << M.getName()
             << (InstrumentAll ? " (all accesses)" : " (unchecked accesses)")
             << "\n";
    }

    declareRuntimeFunctions();

    for (Function &F : M) {
      if (F.isDeclaration())
        continue;
      if (shouldSkipFunction(F))
        continue;
      instrumentFunction(F);
    }

    if (Verbose) {
      errs() << "[FilC] Instrumentation complete: " << NumChecked
             << " checked, " << NumSkipped
             << " skipped (already checked at compile time)\n";
    }
  }

private:
  static bool shouldSkipFunction(Function &F) {
    StringRef Name = F.getName();
    // Avoid instrumenting the FilC runtime or LLVM intrinsics.
    if (Name.starts_with("filc_") || Name.starts_with("llvm."))
      return true;
    return false;
  }

  /// Whether this access still needs a runtime check, or whether rustc already
  /// proved it safe.
  bool needsCheck(Instruction *I) {
    if (InstrumentAll || !I->getMetadata(FilCSafeMD)) {
      ++NumChecked;
      return true;
    }
    ++NumSkipped;
    return false;
  }

  void declareRuntimeFunctions() {
    // void filc_check_read(void* ptr, size_t size, uint32_t expected_type)
    CheckRead =
        M.getOrInsertFunction("filc_check_read", VoidTy, PtrTy, IntPtrTy, Int32Ty);

    // void filc_check_write(void* ptr, size_t size, uint32_t expected_type)
    CheckWrite =
        M.getOrInsertFunction("filc_check_write", VoidTy, PtrTy, IntPtrTy, Int32Ty);

    // void filc_set_type_id(void* ptr, uint32_t type_id)
    SetTypeId = M.getOrInsertFunction("filc_set_type_id", VoidTy, PtrTy, Int32Ty);

    // uint32_t filc_get_type_id(void* ptr)
    GetTypeId = M.getOrInsertFunction("filc_get_type_id", Int32Ty, PtrTy);

    // A failing check aborts rather than unwinding, so the checks never need a
    // landing pad. That matters because they sit in front of accesses in code
    // that can unwind, where an `invoke` per access would be expensive.
    //
    // Nothing stronger is claimed here on purpose. The checks take a lock and
    // read the environment, so they are neither `readonly` nor `willreturn`,
    // and under LTO the runtime's globals are visible to LLVM, which would
    // catch us out if we said otherwise.
    for (FunctionCallee Callee : {CheckRead, CheckWrite, SetTypeId, GetTypeId}) {
      if (auto *F = dyn_cast<Function>(Callee.getCallee()))
        F->setDoesNotThrow();
    }
  }

  uint32_t getTypeId(Type *T) {
    if (!T)
      return 0;

    auto It = TypeIds.find(T);
    if (It != TypeIds.end())
      return It->second;

    uint32_t Id = NextTypeId++;
    TypeIds[T] = Id;

    if (Verbose) {
      errs() << "[FilC] Type ID " << Id << " for " << *T << "\n";
    }

    return Id;
  }

  uint32_t getTypeIdForAccess(Type *T) {
    if (!T)
      return 0;

    if (T->isPointerTy())
      return 0x1000;

    if (isa<IntegerType>(T))
      return getTypeId(T);

    if (T->isFloatTy())
      return 0xF100;
    if (T->isDoubleTy())
      return 0xF200;

    return getTypeId(T);
  }

  void instrumentFunction(Function &F) {
    std::vector<AllocaInst *> AllocsToInstrument;
    std::vector<Instruction *> AccessesToInstrument;

    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        if (auto *AI = dyn_cast<AllocaInst>(&I)) {
          // An unescaped stack slot can only be reached by the accesses to it
          // that are right here in this function, all of which rustc already
          // checked, so the runtime never needs to know about it.
          if (InstrumentAll || allocaMayEscape(AI))
            AllocsToInstrument.push_back(AI);
          continue;
        }

        if (isa<LoadInst>(&I) || isa<StoreInst>(&I) || isa<AtomicRMWInst>(&I) ||
            isa<AtomicCmpXchgInst>(&I) || isa<AnyMemIntrinsic>(&I))
          AccessesToInstrument.push_back(&I);
      }
    }

    if (AllocsToInstrument.empty() && AccessesToInstrument.empty())
      return;

    if (Verbose) {
      errs() << "[FilC] Instrumenting function: " << F.getName() << "\n";
    }

    for (AllocaInst *AI : AllocsToInstrument)
      instrumentAlloca(AI);
    for (Instruction *I : AccessesToInstrument)
      instrumentAccess(I);
  }

  void instrumentAlloca(AllocaInst *AI) {
    if (TrackedAllocas.count(AI))
      return;
    TrackedAllocas.insert(AI);

    Instruction *Next = AI->getNextNode();
    if (!Next)
      return;
    IRBuilder<> Builder(Next);

    Type *AllocTy = AI->getAllocatedType();
    uint32_t TypeId = getTypeId(AllocTy);
    Builder.CreateCall(SetTypeId, {AI, ConstantInt::get(Int32Ty, TypeId)});
  }

  void instrumentAccess(Instruction *I) {
    if (!needsCheck(I))
      return;

    if (auto *LI = dyn_cast<LoadInst>(I)) {
      Type *Ty = LI->getType();
      emitCheck(I, CheckRead, LI->getPointerOperand(),
                DL.getTypeStoreSize(Ty), getTypeIdForAccess(Ty));
      return;
    }

    if (auto *SI = dyn_cast<StoreInst>(I)) {
      Type *Ty = SI->getValueOperand()->getType();
      emitCheck(I, CheckWrite, SI->getPointerOperand(),
                DL.getTypeStoreSize(Ty), getTypeIdForAccess(Ty));
      return;
    }

    // Read-modify-write atomics touch memory both ways, so check the write:
    // it is the stricter of the two permissions.
    if (auto *RMW = dyn_cast<AtomicRMWInst>(I)) {
      Type *Ty = RMW->getValOperand()->getType();
      emitCheck(I, CheckWrite, RMW->getPointerOperand(),
                DL.getTypeStoreSize(Ty), getTypeIdForAccess(Ty));
      return;
    }

    if (auto *CAS = dyn_cast<AtomicCmpXchgInst>(I)) {
      Type *Ty = CAS->getNewValOperand()->getType();
      emitCheck(I, CheckWrite, CAS->getPointerOperand(),
                DL.getTypeStoreSize(Ty), getTypeIdForAccess(Ty));
      return;
    }

    if (auto *MI = dyn_cast<AnyMemIntrinsic>(I)) {
      instrumentMemIntrinsic(MI);
      return;
    }
  }

  /// `memcpy` / `memmove` / `memset` are how `ptr::copy` and friends are
  /// lowered, so they need the same treatment as an explicit store loop.
  void instrumentMemIntrinsic(AnyMemIntrinsic *MI) {
    IRBuilder<> Builder(MI);

    Value *Length = Builder.CreateZExtOrTrunc(MI->getLength(), IntPtrTy);
    Value *UntypedId = ConstantInt::get(Int32Ty, UntypedAccess);

    Builder.CreateCall(CheckWrite, {MI->getRawDest(), Length, UntypedId});

    if (auto *MT = dyn_cast<AnyMemTransferInst>(MI))
      Builder.CreateCall(CheckRead, {MT->getRawSource(), Length, UntypedId});
  }

  void emitCheck(Instruction *At, FunctionCallee Check, Value *Ptr,
                 uint64_t Size, uint32_t ExpectedTypeId) {
    IRBuilder<> Builder(At);
    Builder.CreateCall(Check, {Ptr, ConstantInt::get(IntPtrTy, Size),
                               ConstantInt::get(Int32Ty, ExpectedTypeId)});
  }
};

} // namespace

PreservedAnalyses FilCPass::run(Module &M, ModuleAnalysisManager &MAM) {
  (void)MAM;
  Instrumenter Inst(M, InstrumentAll);
  Inst.run();
  return PreservedAnalyses::none();
}
