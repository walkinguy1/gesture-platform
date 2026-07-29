mod backend;

use std::sync::Mutex;

use tauri::Manager;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .setup(|app| {
            let child = backend::spawn();
            app.manage(backend::BackendProcess(Mutex::new(child)));
            Ok(())
        })
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::CloseRequested { .. } = event {
                if let Some(state) = window.app_handle().try_state::<backend::BackendProcess>() {
                    backend::stop(&state);
                }
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running the Gesture Platform desktop app");
}
