//! Window, event loop, and the interaction model.
//!
//! PLAN.md §17: a simulation is causal, not random-access. There is no seek,
//! so the controls here are play, pause, single-step and reset — and nothing
//! that pretends to be a video scrubber. Rewind arrives with determinism and
//! checkpointing at v0.4.

use std::sync::Arc;
use std::time::Instant;

use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId};

use crate::camera::OrbitCamera;
use crate::mesh::Geometry;
use crate::renderer::Renderer;

/// What the viewport asks of whatever it is displaying.
///
/// The viewport never touches a `World` or a `Simulation` — it takes a scene
/// that knows how to advance itself and how to describe itself as geometry.
/// That is what keeps the dependency pointing one way, and what keeps the
/// simulation core headless (PLAN.md §25).
pub trait Scene {
    /// Advance by one output frame of simulated time.
    fn update(&mut self);
    /// Rebuild the current frame's debug geometry.
    fn draw(&mut self, geometry: &mut Geometry);
    /// Back to t = 0.
    fn reset(&mut self);
    /// Text for the window title bar: the viewport has no font renderer, and
    /// the title is a perfectly good status line.
    fn status(&self) -> String {
        String::new()
    }
    /// Scene-specific keys. The viewport handles the universal ones itself.
    fn on_key(&mut self, _key: char) {}
    /// Suggested initial camera framing.
    fn camera(&self) -> OrbitCamera {
        OrbitCamera::default()
    }
}

pub struct ViewportConfig {
    pub title: String,
    pub width: u32,
    pub height: u32,
    /// Start paused. Useful for demos whose first frame is the interesting one.
    pub start_paused: bool,
}

impl Default for ViewportConfig {
    fn default() -> Self {
        Self {
            title: "Fusion".to_string(),
            width: 1280,
            height: 800,
            start_paused: false,
        }
    }
}

/// Open a window and run `scene` in it until the window closes.
pub fn run(
    scene: impl Scene + 'static,
    config: ViewportConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App {
        scene: Box::new(scene),
        config,
        window: None,
        renderer: None,
        camera: OrbitCamera::default(),
        geometry: Geometry::new(),
        paused: false,
        step_once: false,
        last_title_update: Instant::now(),
        drag: None,
        cursor: (0.0, 0.0),
    };
    app.paused = app.config.start_paused;
    event_loop.run_app(&mut app)?;
    Ok(())
}

#[derive(Clone, Copy, PartialEq)]
enum Drag {
    Orbit,
    Pan,
}

struct App {
    scene: Box<dyn Scene>,
    config: ViewportConfig,
    window: Option<Arc<Window>>,
    renderer: Option<Renderer>,
    camera: OrbitCamera,
    geometry: Geometry,
    paused: bool,
    step_once: bool,
    last_title_update: Instant,
    drag: Option<Drag>,
    cursor: (f32, f32),
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attributes = Window::default_attributes()
            .with_title(&self.config.title)
            .with_inner_size(PhysicalSize::new(self.config.width, self.config.height));
        let window = Arc::new(event_loop.create_window(attributes).expect("create window"));
        self.camera = self.scene.camera();
        self.renderer = Some(Renderer::new(window.clone()));
        self.window = Some(window);
        println!(
            "controls: [space] pause  [.] single step  [r] reset  \
             drag = orbit  shift-drag = pan  scroll = zoom  [esc] quit"
        );
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::Resized(size) => {
                if let Some(r) = &mut self.renderer {
                    r.resize(size.width, size.height);
                }
            }

            WindowEvent::KeyboardInput { event, .. } if event.state == ElementState::Pressed => {
                match event.logical_key.as_ref() {
                    Key::Named(NamedKey::Escape) => event_loop.exit(),
                    Key::Named(NamedKey::Space) => self.paused = !self.paused,
                    Key::Character(".") => {
                        self.step_once = true;
                        self.paused = true;
                    }
                    Key::Character("r") | Key::Character("R") => {
                        self.scene.reset();
                    }
                    Key::Character(c) => {
                        if let Some(ch) = c.chars().next() {
                            self.scene.on_key(ch);
                        }
                    }
                    _ => {}
                }
            }

            WindowEvent::MouseInput { state, button, .. } => {
                let pressed = state == ElementState::Pressed;
                match button {
                    MouseButton::Left => {
                        self.drag = pressed.then_some(Drag::Orbit);
                    }
                    MouseButton::Right | MouseButton::Middle => {
                        self.drag = pressed.then_some(Drag::Pan);
                    }
                    _ => {}
                }
            }

            WindowEvent::CursorMoved { position, .. } => {
                let (x, y) = (position.x as f32, position.y as f32);
                let (dx, dy) = (x - self.cursor.0, y - self.cursor.1);
                self.cursor = (x, y);
                match self.drag {
                    Some(Drag::Orbit) => self.camera.orbit(dx * 0.006, dy * 0.006),
                    Some(Drag::Pan) => self.camera.pan(dx, dy),
                    None => {}
                }
            }

            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32 * 0.02,
                };
                self.camera.zoom((1.0 - scroll * 0.1).clamp(0.5, 2.0));
            }

            WindowEvent::RedrawRequested => self.frame(),

            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(w) = &self.window {
            w.request_redraw();
        }
    }
}

impl App {
    fn frame(&mut self) {
        if !self.paused || self.step_once {
            self.scene.update();
            self.step_once = false;
        }

        self.geometry.clear();
        self.scene.draw(&mut self.geometry);

        if let Some(r) = &mut self.renderer {
            r.render(&self.camera, &self.geometry);
        }

        // Retitling every frame is a surprisingly expensive syscall on some
        // platforms, so status updates are throttled.
        if self.last_title_update.elapsed().as_millis() > 100 {
            self.last_title_update = Instant::now();
            if let Some(w) = &self.window {
                let state = if self.paused { "paused" } else { "running" };
                w.set_title(&format!(
                    "{} — {} — {}",
                    self.config.title,
                    state,
                    self.scene.status()
                ));
            }
        }
    }
}
