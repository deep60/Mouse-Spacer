use enigo::{Enigo, KeyboardControllable, MouseControllable}; // For mouse/keyboard control
use mediapipe_rs::{hands::HandLandmarker, tasks::vision::RunningMode}; // Hypothetical MediaPipe Rust binding
use nokhwa::{Camera, CameraFormat, FrameFormat}; // For webcam capture
use opencv::{
    core::{flip, Mat, Point, Scalar, Vector},
    imgproc::{circle, line},
    prelude::*,
    videoio,
};
use std::error::Error;
use tch::{nn, Device, Tensor}; // PyTorch bindings for Rust

#[derive(Debug)]
struct HandGestureController {
    camera: Camera,
    hand_landmarker: HandLandmarker,
    model: nn::Module,
    enigo: Enigo,
    status: i32,
    last_distance: f32,
    last_coord_x: i32,
    last_coord_y: i32,
    is_clicked: bool,
}

impl HandGestureController {
    fn new() -> Result<Self, Box<dyn Error>> {
        //Initialize Camera
        let camera = Camera::new(0, CameraFormat::new_from(640, 480, FrameFront::MJPEG, 30))?;

        // Initialize Mediapipe hand landmarker
        let hand_landmarker = HandLandmarker::new(RunningMode::LiveStream, 1, 0.7)?;
        // Load the model
        let model = tch::CModule::load("./my_model/model.pt")?;

        Ok(HandGestureController {
            camera,
            hand_landmarker,
            model,
            enigo: Enigo::new(),
            status: -1,
            last_distance: -1.0,
            last_coord_x: 0,
            last_coord_y: 0,
            is_clicked: false,
        })
    }

    fn calculate_distances(&self, point1: Point, point2: Point) -> f32 {
        let dx = point1.x - point2.x;
        let dy = point1.y - point2.y;
        ((dx * dx + dy * dy) as f32).sqrt()
    }

    fn process_frame(&mut self) -> Result<bool, Box<dyn Error>> {
        let frame = self.camera.frame()?;
        let mut image = Mat::from_slices(&frame)?;
        flip(&image, &mut image, 1)?;

        //process hand landmarks
        if let Some(hand_landmarks) = self.hand_landmarker.detect(&image)?.first() {
            let (h, w) = (image.rows()?, image.cols()?);

            // Extract landmark points
            let point_a = Point::new(
                (hand_landmarks[4].x * w as f32) as i32,
                (hand_landmarks[4].y * h as f32) as i32,
            );
            let point_b = Point::new(
                (hand_landmarks[8].x * w as f32) as i32,
                (hand_landmarks[8].y * h as f32) as i32,
            );
            let point_c = Point::new(
                (hand_landmarks[12].x * w as f32) as i32,
                (hand_landmarks[12].y * h as f32) as i32,
            );

            // Prepare landmarks for model prediction
            let mut pose_array = Vec::new();
            for landmark in hand_landmarks {
                pose_array.push(landmark.x);
                pose_array.push(landmark.y);
                pose_array.push(landmark.z);
            }

            // Model prediction
            let input_tensor = Tensor::from_slice(&pose_array).view((-1, 3)).unsqueeze(0);
            let prediction = self.model.forward_t(&input_tensor, false)?;
            let state = prediction.argmax(1, false).int64_value(&[0]) as i32;
            let confidence = prediction.double_value(&[0, state as i64]);

            if confidence > 0.95 {
                match state {
                    0 => self.handle_state_zero(point_a, point_b)?,
                    1 => self.handle_state_one(point_a, point_b, point_c)?,
                    _ => {}
                }
            }
        }

        Ok(true)
    }

    fn handle_state_zero(&mut self, point_a: Point, point_b: Point) -> Result<(), Box<dyn Error>> {
        let distance_ab = self.calculate_distances(point_a, point_b) / 10.0;

        if distance_ab < 8.0 && !self.is_clicked {
            self.status = 1; //CLICKED_DOWN
            let (screen_w, screen_h) = self.enigo.main_display_size();
            self.enigo.move_mouse_to(screen_w / 2, screen_h / 2);
            self.enigo.mouse_down(enigo::MouseButton::Right);
            self.enigo.key_down(enigo::Key::Control);
            self.last_coord_x = point_a.x;
            self.last_coord_y = point_a.y;
            self.is_clicked = true;
        }

        if distance_ab < 12.0 && self.is_clicked {
            self.status = 2;
            self.enigo.key_up(enigo::Key::Control);
            self.enigo.mouse_up(enigo::MouseButton::Right);
            self.last_coord_x = point_a.x;
            self.last_coord_y = point_a.y;
            self.is_clicked = false;
        }

        //Handle Mouse Movement
        if distance_ab < 15.0
            && (self.last_coord_x - point_a.x).abs() > 5
            && (self.last_coord_y - point_a.y).abs() > 5
        {
            let dx = point_a.x - self.last_coord_x;
            let dy = point_a.y - self.last_coord_y;

            if self.status == 1 {
                let (mouse_x, mouse_y) = self.enigo.mouse_location();
                self.enigo.move_mouse_to(mouse_x + dx, mouse_y + dy);
            }

            self.last_coord_x = point_a.x;
            self.last_coord_y = point_a.y;
        }

        Ok(())
    }

    fn handle_state_one(
        &mut self,
        point_a: Point,
        point_b: Point,
        point_c: Point,
    ) -> Result<(), Box<dyn Error>> {
        let distance_ab = self.calculate_distances(point_a, point_b);
        let distance_bc = self.calculate_distances(point_b, point_c);
        let distance_ca = self.calculate_distances(point_c, point_a);

        let total_distance = distance_ab + distance_bs + distance_ca;
        let total_distance_normalized = total_distances / 10.0;

        if self.last_distance == -1.0 {
            self.last_distance = total_distance_normalized;
        }

        let delta_scroll = total_distance_normalized - self.last_distance;
        self.last_distance = total_distance_normalized;

        if delta_scroll > 1.0 && delta_scroll < 30.0 {
            self.enigo.mouse_scroll_y(delta_scroll as i32 / 5);
        }

        if delta_scroll < -1.0 && delta_scroll > -30.0 {
            self.enigo.mouse_scroll_y(delta_scroll as i32 / 5);
        }

        Ok(())
    }

    fn run(&mut self) -> Result<(), Box<dyn Error>> {
        while self.process_frame()? {
            if opencv::highgui::wait_key(1)? == 'q' as i32 {
                break;
            }
        }

        Ok(())
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut controller = HandGestureController::new()?;
    controller.run()?;
    Ok(())
}
