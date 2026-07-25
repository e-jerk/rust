// Optimized runtime for FilC memory safety pass
//
// Uses ARM MTE (Memory Tagging Extension) when available for near-zero overhead.
// Falls back to hash table-based software checks on systems without MTE.
//
// Architecture:
//   - MTE path: Hardware catches UAF/bounds, software only checks metadata
//   - Software path: O(1) hash table lookup for bounds/UAF/metadata
//
// Key optimizations:
//   1. Hash table (O(1)) instead of linear search (O(n))
//   2. MTE hardware acceleration for bounds/UAF on ARM64
//   3. Minimal check functions - only metadata (type, readonly)
//   4. Tagged pointers for instant freed-status detection

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdbool.h>
#include <sys/mman.h>
#include <dlfcn.h>
#include <pthread.h>
#include <unistd.h>
#include <errno.h>

#ifdef __aarch64__
#ifdef __linux__
#include <sys/prctl.h>
#include <asm/hwcap.h>
#include <sys/auxv.h>
#endif
#endif

// === Configuration ===

// Hash table size (power of 2 for fast modulo)
#define HASH_BITS 14
#define HASH_SIZE (1 << HASH_BITS)
#define HASH_MASK (HASH_SIZE - 1)

// Allocation metadata
struct filc_alloc_meta {
    void* base;
    size_t size;
    uint32_t type_id;
    bool freed;
    bool readonly;
    uint8_t mte_tag;       // MTE tag (0-15), 0xff = untagged
    struct filc_alloc_meta* next;  // Hash chain
};

// Hash table buckets (each bucket is a linked list head)
static struct filc_alloc_meta* g_hash_table[HASH_SIZE];
static pthread_rwlock_t g_hash_lock = PTHREAD_RWLOCK_INITIALIZER;
static _Atomic int g_alloc_count = 0;

// MTE state
static bool g_mte_available = false;
static bool g_mte_enabled = false;

// Cached real libc functions
static void* (*real_malloc)(size_t) = NULL;
static void (*real_free)(void*) = NULL;
static void* (*real_calloc)(size_t, size_t) = NULL;
static void* (*real_realloc)(void*, size_t) = NULL;
static void* (*real_mmap)(void*, size_t, int, int, int, off_t) = NULL;
static int (*real_munmap)(void*, size_t) = NULL;
static int real_functions_initialized = 0;

// === MTE Support ===

#ifdef __aarch64__

// Check if CPU supports MTE
static bool check_mte_support(void) {
    #ifdef __linux__
    unsigned long hwcap = getauxval(AT_HWCAP2);
    return (hwcap & HWCAP2_MTE) != 0;
    #else
    return false;  // macOS and other platforms don't expose MTE to userspace yet
    #endif
}

// Enable MTE for this process
static bool enable_mte(void) {
    #if defined(__linux__) && defined(PR_SET_TAGGED_ADDR_CTRL)
    int ret = prctl(PR_SET_TAGGED_ADDR_CTRL,
                    PR_TAGGED_ADDR_ENABLE | PR_MTE_TCF_SYNC | (0xfffe << PR_MTE_TAG_SHIFT),
                    0, 0, 0);
    if (ret < 0) {
        return false;
    }
    return true;
    #else
    return false;
    #endif
}

// Get random MTE tag (1-15, avoid 0)
static inline uint8_t random_tag(void) {
    // Simple LCG - good enough for tags
    static uint32_t seed = 123456789;
    seed = seed * 1103515245 + 12345;
    return (seed % 15) + 1;  // 1-15
}

// Set MTE tags for a memory range using inline assembly
// Uses STG (store tag) instruction
static void mte_set_tags(void* ptr, size_t size, uint8_t tag) {
    if (!g_mte_enabled || !ptr) return;
    
    #ifdef __linux__
    // Tag is in bits 56-59 of the address we pass to STG
    uintptr_t tagged = ((uintptr_t)ptr & ~((uintptr_t)0xF << 56)) | ((uintptr_t)tag << 56);
    
    size_t granules = (size + 15) / 16;
    void* current = (void*)tagged;
    
    for (size_t i = 0; i < granules; i++) {
        __asm__ volatile (
            "stg %0, [%0]"
            :
            : "r"(current)
            : "memory"
        );
        current = (char*)current + 16;
    }
    
    __asm__ volatile ("dsb ish" ::: "memory");
    __asm__ volatile ("isb" ::: "memory");
    #else
    (void)size;
    (void)tag;
    #endif
}

// Tag a pointer with MTE tag
static inline void* tag_pointer(void* ptr, uint8_t tag) {
    if (!g_mte_enabled || !ptr) return ptr;
    return (void*)(((uintptr_t)ptr & ~((uintptr_t)0xF << 56)) | ((uintptr_t)tag << 56));
}

// Untag a pointer (clear bits 56-59)
static inline void* untag_pointer(void* ptr) {
    return (void*)((uintptr_t)ptr & ~((uintptr_t)0xF << 56));
}

#else

static bool check_mte_support(void) { return false; }
static bool enable_mte(void) { return false; }
static inline uint8_t random_tag(void) { return 0; }
static void mte_set_tags(void* ptr, size_t size, uint8_t tag) { (void)ptr; (void)size; (void)tag; }
static inline void* tag_pointer(void* ptr, uint8_t tag) { (void)tag; return ptr; }
static inline void* untag_pointer(void* ptr) { return ptr; }

#endif

// === Panic Handler ===

static void filc_panic(const char* msg) {
    fprintf(stderr, "filc safety error: %s\n", msg);
    abort();
}

// === Hash Table ===

// Fast hash function for pointers
static inline uint32_t hash_ptr(void* ptr) {
    uintptr_t p = (uintptr_t)ptr;
    // FNV-1a inspired hash
    uint32_t hash = 2166136261u;
    hash ^= (uint32_t)(p & 0xFFFFFFFF);
    hash *= 16777619;
    hash ^= (uint32_t)(p >> 32);
    hash *= 16777619;
    return hash;
}

// Get bucket index for a pointer
static inline uint32_t bucket_idx(void* ptr) {
    return hash_ptr(ptr) & HASH_MASK;
}

// Find metadata in hash table (read lock held)
static struct filc_alloc_meta* find_meta_unlocked(void* ptr) {
    if (!ptr) return NULL;
    
    void* untagged = untag_pointer(ptr);
    uint32_t idx = bucket_idx(untagged);
    
    for (struct filc_alloc_meta* meta = g_hash_table[idx]; meta; meta = meta->next) {
        if (meta->base == untagged) {
            return meta;
        }
    }
    return NULL;
}

// Find exact metadata including freed entries
static struct filc_alloc_meta* find_exact_meta_unlocked(void* ptr) {
    return find_meta_unlocked(ptr);
}

// Check if address was ever freed
static bool is_freed_address_unlocked(void* ptr) {
    struct filc_alloc_meta* meta = find_meta_unlocked(ptr);
    return meta && meta->freed;
}

// Insert metadata into hash table (write lock held)
static void insert_meta_unlocked(struct filc_alloc_meta* meta) {
    uint32_t idx = bucket_idx(meta->base);
    meta->next = g_hash_table[idx];
    g_hash_table[idx] = meta;
    g_alloc_count++;
}

// Mark meta as freed (write lock held)
static void mark_freed_unlocked(void* ptr) {
    struct filc_alloc_meta* meta = find_meta_unlocked(ptr);
    if (meta && !meta->freed) {
        meta->freed = true;
        
        // With MTE: retag with new random tag so old pointers mismatch
        if (g_mte_enabled && meta->mte_tag != 0xff) {
            uint8_t new_tag = random_tag();
            meta->mte_tag = new_tag;
            mte_set_tags(meta->base, meta->size, new_tag);
        }
    }
}

// === Allocation Registration ===

static void register_alloc(void* ptr, size_t size, uint32_t type_id, bool readonly) {
    if (!ptr) return;
    
    struct filc_alloc_meta* meta = real_malloc(sizeof(struct filc_alloc_meta));
    if (!meta) {
        filc_panic("allocation metadata allocation failed");
    }
    
    meta->base = untag_pointer(ptr);
    meta->size = size;
    meta->type_id = type_id;
    meta->freed = false;
    meta->readonly = readonly;
    
    if (g_mte_enabled) {
        meta->mte_tag = random_tag();
        mte_set_tags(meta->base, meta->size, meta->mte_tag);
    } else {
        meta->mte_tag = 0xff;
    }
    
    pthread_rwlock_wrlock(&g_hash_lock);
    insert_meta_unlocked(meta);
    pthread_rwlock_unlock(&g_hash_lock);
}

// === Check Functions ===

void filc_check_read(void* ptr, size_t size, uint32_t expected_type) {
    (void)size;  // MTE or segfault handles bounds
    
    if (!ptr) {
        // Null pointer - let hardware segfault or catch it
        // Only panic if we're not using MTE (more helpful error)
        if (!g_mte_enabled) {
            filc_panic("cannot access null pointer");
        }
        return;
    }
    
    // With MTE: hardware catches UAF and bounds via tag mismatch
    // Without MTE: we need software checks
    if (!g_mte_enabled) {
        pthread_rwlock_rdlock(&g_hash_lock);
        
        if (is_freed_address_unlocked(ptr)) {
            pthread_rwlock_unlock(&g_hash_lock);
            filc_panic("cannot read pointer with invalid capability (use-after-free)");
        }
        
        struct filc_alloc_meta* meta = find_meta_unlocked(ptr);
        if (meta) {
            if (meta->freed) {
                pthread_rwlock_unlock(&g_hash_lock);
                filc_panic("cannot read pointer with invalid capability (use-after-free)");
            }
            
            void* end = (char*)ptr + size;
            void* alloc_end = (char*)meta->base + meta->size;
            if (end > alloc_end) {
                pthread_rwlock_unlock(&g_hash_lock);
                filc_panic("cannot read pointer with ptr >= upper (buffer overflow)");
            }
        }
        pthread_rwlock_unlock(&g_hash_lock);
    }
    
    // Type check (software, always needed)
    if (expected_type != 0) {
        pthread_rwlock_rdlock(&g_hash_lock);
        struct filc_alloc_meta* meta = find_meta_unlocked(ptr);
        if (meta && meta->type_id != 0 && meta->type_id != expected_type) {
            pthread_rwlock_unlock(&g_hash_lock);
            const char* disable = getenv("FILC_DISABLE_TYPE_CHECK");
            if (!disable) {
                filc_panic("type confusion - accessing memory with wrong type");
            }
        } else {
            pthread_rwlock_unlock(&g_hash_lock);
        }
    }
}

void filc_check_write(void* ptr, size_t size, uint32_t expected_type) {
    (void)size;  // MTE or segfault handles bounds
    
    if (!ptr) {
        if (!g_mte_enabled) {
            filc_panic("cannot access null pointer");
        }
        return;
    }
    
    // Software fallback for non-MTE systems
    if (!g_mte_enabled) {
        pthread_rwlock_rdlock(&g_hash_lock);
        
        if (is_freed_address_unlocked(ptr)) {
            pthread_rwlock_unlock(&g_hash_lock);
            filc_panic("cannot write pointer with invalid capability (use-after-free)");
        }
        
        struct filc_alloc_meta* meta = find_meta_unlocked(ptr);
        if (meta) {
            if (meta->freed) {
                pthread_rwlock_unlock(&g_hash_lock);
                filc_panic("cannot write pointer with invalid capability (use-after-free)");
            }
            
            if (meta->readonly) {
                pthread_rwlock_unlock(&g_hash_lock);
                filc_panic("cannot write pointer with readonly capability");
            }
            
            void* end = (char*)ptr + size;
            void* alloc_end = (char*)meta->base + meta->size;
            if (end > alloc_end) {
                pthread_rwlock_unlock(&g_hash_lock);
                filc_panic("cannot write pointer with ptr >= upper (buffer overflow)");
            }
        }
        pthread_rwlock_unlock(&g_hash_lock);
    } else {
        // MTE enabled: only need to check readonly + type
        pthread_rwlock_rdlock(&g_hash_lock);
        struct filc_alloc_meta* meta = find_meta_unlocked(ptr);
        if (meta && meta->readonly) {
            pthread_rwlock_unlock(&g_hash_lock);
            filc_panic("cannot write pointer with readonly capability");
        }
        pthread_rwlock_unlock(&g_hash_lock);
    }
    
    // Type check
    if (expected_type != 0) {
        pthread_rwlock_rdlock(&g_hash_lock);
        struct filc_alloc_meta* meta = find_meta_unlocked(ptr);
        if (meta && meta->type_id != 0 && meta->type_id != expected_type) {
            pthread_rwlock_unlock(&g_hash_lock);
            const char* disable = getenv("FILC_DISABLE_TYPE_CHECK");
            if (!disable) {
                filc_panic("type confusion - accessing memory with wrong type");
            }
        } else {
            pthread_rwlock_unlock(&g_hash_lock);
        }
    }
}

void filc_check_not_freed(void* ptr) {
    if (!ptr) return;
    
    if (!g_mte_enabled) {
        pthread_rwlock_rdlock(&g_hash_lock);
        bool freed = is_freed_address_unlocked(ptr);
        pthread_rwlock_unlock(&g_hash_lock);
        
        if (freed) {
            filc_panic("cannot access pointer with invalid capability (use-after-free)");
        }
    }
    // With MTE: hardware catches this
}

void filc_set_type_id(void* ptr, uint32_t type_id) {
    if (!ptr) return;
    
    pthread_rwlock_wrlock(&g_hash_lock);
    struct filc_alloc_meta* meta = find_meta_unlocked(ptr);
    if (meta) {
        meta->type_id = type_id;
    }
    pthread_rwlock_unlock(&g_hash_lock);
}

uint32_t filc_get_type_id(void* ptr) {
    if (!ptr) return 0;
    
    pthread_rwlock_rdlock(&g_hash_lock);
    struct filc_alloc_meta* meta = find_meta_unlocked(ptr);
    uint32_t result = meta ? meta->type_id : 0;
    pthread_rwlock_unlock(&g_hash_lock);
    return result;
}

// === Function Interposition ===

static void init_real_functions(void) {
    if (real_functions_initialized) return;
    real_functions_initialized = 1;
    
    real_malloc = dlsym(RTLD_NEXT, "malloc");
    real_free = dlsym(RTLD_NEXT, "free");
    real_calloc = dlsym(RTLD_NEXT, "calloc");
    real_realloc = dlsym(RTLD_NEXT, "realloc");
    real_mmap = dlsym(RTLD_NEXT, "mmap");
    real_munmap = dlsym(RTLD_NEXT, "munmap");
    
    // Initialize MTE
    #ifdef __aarch64__
    if (check_mte_support()) {
        if (enable_mte()) {
            g_mte_available = true;
            g_mte_enabled = true;
            fprintf(stderr, "[filc] MTE enabled - using hardware-accelerated memory safety\n");
        } else {
            fprintf(stderr, "[filc] MTE supported but could not enable (try: echo 1 > /proc/sys/kernel/arm64.tagged_addr.enabled)\n");
        }
    } else {
        fprintf(stderr, "[filc] MTE not available - using software checks (expect ~5x overhead)\n");
    }
    #else
    fprintf(stderr, "[filc] Running on non-ARM64 - using software checks (expect ~5x overhead)\n");
    #endif
}

void* malloc(size_t size) {
    init_real_functions();
    void* ptr = real_malloc(size);
    if (ptr) {
        register_alloc(ptr, size, 0, false);
        if (g_mte_enabled) {
            pthread_rwlock_rdlock(&g_hash_lock);
            struct filc_alloc_meta* meta = find_meta_unlocked(ptr);
            uint8_t tag = meta ? meta->mte_tag : 0;
            pthread_rwlock_unlock(&g_hash_lock);
            return tag_pointer(ptr, tag);
        }
    }
    return ptr;
}

void free(void* ptr) {
    init_real_functions();
    if (ptr) {
        void* untagged = untag_pointer(ptr);
        pthread_rwlock_wrlock(&g_hash_lock);
        
        struct filc_alloc_meta* meta = find_meta_unlocked(untagged);
        if (meta) {
            if (meta->freed) {
                pthread_rwlock_unlock(&g_hash_lock);
                filc_panic("cannot free pointer with invalid capability (double free)");
            }
            meta->freed = true;
            
            // Retag with new random tag so old pointers mismatch
            if (g_mte_enabled && meta->mte_tag != 0xff) {
                uint8_t new_tag = random_tag();
                meta->mte_tag = new_tag;
                mte_set_tags(meta->base, meta->size, new_tag);
            }
        }
        pthread_rwlock_unlock(&g_hash_lock);
        real_free(untagged);
    }
}

void* calloc(size_t nmemb, size_t size) {
    init_real_functions();
    void* ptr = real_calloc(nmemb, size);
    if (ptr) {
        register_alloc(ptr, nmemb * size, 0, false);
        if (g_mte_enabled) {
            pthread_rwlock_rdlock(&g_hash_lock);
            struct filc_alloc_meta* meta = find_meta_unlocked(ptr);
            uint8_t tag = meta ? meta->mte_tag : 0;
            pthread_rwlock_unlock(&g_hash_lock);
            return tag_pointer(ptr, tag);
        }
    }
    return ptr;
}

void* realloc(void* ptr, size_t size) {
    init_real_functions();
    
    void* untagged = untag_pointer(ptr);
    
    // Mark old allocation as freed
    if (untagged) {
        pthread_rwlock_wrlock(&g_hash_lock);
        mark_freed_unlocked(untagged);
        pthread_rwlock_unlock(&g_hash_lock);
    }
    
    void* new_ptr = real_realloc(untagged, size);
    if (new_ptr) {
        register_alloc(new_ptr, size, 0, false);
        if (g_mte_enabled) {
            pthread_rwlock_rdlock(&g_hash_lock);
            struct filc_alloc_meta* meta = find_meta_unlocked(new_ptr);
            uint8_t tag = meta ? meta->mte_tag : 0;
            pthread_rwlock_unlock(&g_hash_lock);
            return tag_pointer(new_ptr, tag);
        }
    }
    return new_ptr;
}

void* mmap(void* addr, size_t length, int prot, int flags, int fd, off_t offset) {
    init_real_functions();
    
    #if defined(__aarch64__) && defined(__linux__) && defined(PROT_MTE)
    // Enable MTE for this mapping if MTE is active
    if (g_mte_enabled) {
        prot |= PROT_MTE;
    }
    #endif
    
    void* ptr = real_mmap(addr, length, prot, flags, fd, offset);
    if (ptr != MAP_FAILED) {
        bool readonly = !(prot & PROT_WRITE);
        register_alloc(ptr, length, 0, readonly);
        if (g_mte_enabled) {
            pthread_rwlock_rdlock(&g_hash_lock);
            struct filc_alloc_meta* meta = find_meta_unlocked(ptr);
            uint8_t tag = meta ? meta->mte_tag : 0;
            pthread_rwlock_unlock(&g_hash_lock);
            return tag_pointer(ptr, tag);
        }
    }
    return ptr;
}

int munmap(void* addr, size_t length) {
    init_real_functions();
    if (addr) {
        void* untagged = untag_pointer(addr);
        pthread_rwlock_wrlock(&g_hash_lock);
        
        struct filc_alloc_meta* meta = find_meta_unlocked(untagged);
        if (meta) {
            if (meta->freed) {
                pthread_rwlock_unlock(&g_hash_lock);
                filc_panic("cannot free pointer with invalid capability (double free)");
            }
            meta->freed = true;
            
            if (g_mte_enabled && meta->mte_tag != 0xff) {
                uint8_t new_tag = random_tag();
                meta->mte_tag = new_tag;
                mte_set_tags(meta->base, meta->size, new_tag);
            }
        }
        pthread_rwlock_unlock(&g_hash_lock);
        return real_munmap(untagged, length);
    }
    return real_munmap(addr, length);
}

// === Legacy exported functions (for compatibility) ===

void* filc_tracked_malloc(size_t size, uint32_t type_id) {
    return malloc(size);  // Interposition handles tracking
}

void filc_tracked_free(void* ptr) {
    free(ptr);
}

void* filc_tracked_mmap(void* addr, size_t length, int prot, int flags, int fd, int64_t offset) {
    return mmap(addr, length, prot, flags, fd, (off_t)offset);
}

void filc_tracked_munmap(void* ptr, size_t length) {
    munmap(ptr, length);
}
