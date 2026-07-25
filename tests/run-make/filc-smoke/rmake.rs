// Smoke test for `-Z fil-c`: compile and run a tiny program with Fil-C
// instrumentation and the Fil-C runtime. Also checks that a use-after-free
// is caught by the runtime.
//@ only-unix
//@ ignore-cross-compile

use run_make_support::{run, run_fail, rustc};

fn main() {
    rustc().input("main.rs").arg("-Zfil-c").run();
    run("main").assert_stdout_contains("ok");

    rustc().input("uaf.rs").arg("-Zfil-c").run();
    run_fail("uaf").assert_stderr_contains("filc safety error");
}
