// Launches and tears down the Python recognizer backend (scripts/realtime_demo.py
// --headless) as a child process of the desktop app, so the user doesn't have to
// start it manually in a separate terminal before the WebSocket bridge has anything
// to connect to.
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;

pub struct BackendProcess(pub Mutex<Option<Child>>);

/// Repo root, resolved from the compile-time manifest path (`apps/desktop/src-tauri`
/// on the machine that built this binary). Only correct for local dev builds run out
/// of this checkout -- a distributed build would need the Python backend bundled as
/// a proper sidecar instead of launched from source.
fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("..")
}

fn python_executable(root: &PathBuf) -> PathBuf {
    #[cfg(windows)]
    let venv_python = root.join("venv").join("Scripts").join("python.exe");
    #[cfg(not(windows))]
    let venv_python = root.join("venv").join("bin").join("python");

    if venv_python.exists() {
        venv_python
    } else {
        // Fall back to whatever "python" resolves to on PATH so this still
        // works on a machine where the venv lives somewhere else.
        PathBuf::from("python")
    }
}

/// Spawn the headless recognizer backend. Returns `None` (logging why) instead of
/// panicking if it can't start -- the app is still usable without live predictions,
/// and `useBridge.js` already reconnects with backoff once the backend does come up.
pub fn spawn() -> Option<Child> {
    let root = repo_root();
    let python = python_executable(&root);
    let script = root.join("scripts").join("realtime_demo.py");

    if !script.exists() {
        eprintln!(
            "[backend] {} not found; skipping Python backend launch.",
            script.display()
        );
        return None;
    }

    match Command::new(&python)
        .arg(&script)
        .arg("--headless")
        .arg("--smoothing")
        .current_dir(&root)
        .stdin(Stdio::null())
        .spawn()
    {
        Ok(child) => {
            println!(
                "[backend] launched {} (pid {})",
                python.display(),
                child.id()
            );
            Some(child)
        }
        Err(err) => {
            eprintln!(
                "[backend] failed to launch Python recognizer backend ({}): {err}. \
                 Start it manually with `python scripts/realtime_demo.py --ws-bridge` \
                 if you want live predictions.",
                python.display()
            );
            None
        }
    }
}

/// Best-effort shutdown: a forceful kill, so the camera/model resources are freed
/// even though Python's `finally` cleanup in `run_headless()` won't get to run.
pub fn stop(state: &BackendProcess) {
    if let Ok(mut guard) = state.0.lock() {
        if let Some(mut child) = guard.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}
